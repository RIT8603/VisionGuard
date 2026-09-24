"""RTSP CCTV / IP camera source - the production input.

Status: implemented (Phase 5).

An RTSP stream fails in ways a webcam and a file never do: the network drops,
the camera reboots, a switch buffers, credentials rotate. Three things here
address that.

**Freshness over completeness.** A background thread reads the socket
continuously and keeps only the newest frame. Without it, a pipeline that
processes slower than the stream arrives (14 fps of inference against a 25 fps
camera) falls further behind every second, and the "live" view drifts minutes
into the past. VisionGuard drops stale frames instead. If you need every frame,
record to disk and use the file source.

**Reconnect with backoff.** A dropped stream reopens automatically, with the
delay doubling each consecutive failure so a rebooting camera is not hammered.

**Timeouts.** A dead camera that accepts a TCP connection and then says nothing
must not block the pipeline forever.

Nothing here is visible to the AI modules: an RTSP camera produces exactly the
same :class:`~app.sources.base.FramePacket` a webcam does.
"""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from app.core.config import CaptureBackend, VideoConfig
from app.sources.base import FramePacket, FrameReadError, SourceOpenError
from app.sources.opencv_base import CaptureTarget, OpenCVVideoSource
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: OpenCV reads this at VideoCapture construction to configure FFmpeg.
_FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def redact_url(url: str) -> str:
    """Hide the password in an RTSP URL so it is safe to log.

    ``rtsp://admin:hunter2@10.0.0.5/stream`` -> ``rtsp://admin:***@10.0.0.5/stream``

    Credentials in camera URLs are routinely real and routinely reused, so no
    code path may ever write one to a log file or an error message.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        # Unparseable: fall back to a blunt regex rather than risk leaking.
        return re.sub(r"//[^/@]*@", "//***@", url)

    # A password can be present without a username (rtsp://:secret@host),
    # so both must be checked - testing only username leaks that form.
    if not parts.username and not parts.password:
        return url

    userinfo = parts.username or ""
    if parts.password:
        userinfo += ":***"

    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"

    return urlunsplit(
        (parts.scheme, f"{userinfo}@{host}", parts.path, parts.query, parts.fragment)
    )


class _LatestFrameReader:
    """Drains a capture on its own thread, keeping only the newest frame.

    This is the piece that keeps live latency bounded. The reader never blocks
    on the consumer; if the pipeline is slow, frames are discarded, and the
    count is reported so the drop rate is visible rather than silent.
    """

    def __init__(self, capture: "cv2.VideoCapture", name: str) -> None:
        self._capture = capture
        self._name = name
        self._condition = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._failed = False
        self._dropped = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"rtsp-reader-{self._name}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._capture.read()
            except cv2.error as exc:  # pragma: no cover - driver specific
                logger.debug("Reader thread capture error: %s", exc)
                ok, frame = False, None

            if not ok or frame is None:
                with self._condition:
                    self._failed = True
                    self._condition.notify_all()
                return

            with self._condition:
                if self._frame is not None:
                    self._dropped += 1
                self._frame = frame
                self._condition.notify_all()

    def latest(self, timeout: float) -> Optional[np.ndarray]:
        """Take the newest frame, waiting up to ``timeout`` seconds."""
        deadline = time.perf_counter() + max(0.0, timeout)
        with self._condition:
            while self._frame is None and not self._failed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            frame = self._frame
            self._frame = None
            return frame

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failed and self._frame is None

    @property
    def frames_dropped(self) -> int:
        with self._condition:
            return self._dropped

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            # The thread may be parked inside a blocking read; it is a daemon
            # and the capture is about to be released, so do not wait forever.
            self._thread.join(timeout=2.0)
        self._thread = None


class RTSPSource(OpenCVVideoSource):
    """Reads frames from an RTSP stream."""

    source_type = "rtsp"

    def __init__(self, config: VideoConfig, source_id: str | None = None) -> None:
        super().__init__(source_id or config.resolved_source_id(), config)
        self._reader: Optional[_LatestFrameReader] = None
        self._reconnects = 0
        self._total_reconnects = 0
        self._total_dropped = 0

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def url(self) -> str:
        return self._config.rtsp_url or ""

    @property
    def safe_url(self) -> str:
        """The URL with its password masked. Use this in every message."""
        return redact_url(self.url)

    @property
    def total_reconnects(self) -> int:
        return self._total_reconnects

    @property
    def frames_dropped(self) -> int:
        """Frames discarded to stay live. A high number means slow processing."""
        dropped = self._total_dropped
        if self._reader is not None:
            dropped += self._reader.frames_dropped
        return dropped

    def _capture_target(self) -> CaptureTarget:
        return self.url

    def _resolve_backend(self) -> CaptureBackend:
        if self._config.backend is not CaptureBackend.AUTO:
            return self._config.backend
        return CaptureBackend.FFMPEG

    def _describe(self) -> str:
        return f"RTSP {self.safe_url}"

    def _loggable_target(self) -> str:
        """Never let the raw URL reach a log line."""
        return self.safe_url

    def _open_failure_message(self, target: CaptureTarget, backend: CaptureBackend) -> str:
        return (
            f"Could not open RTSP stream {self.safe_url} "
            f"(transport={self._config.rtsp_transport}).\n"
            "Check that:\n"
            "  - the camera is reachable from this host\n"
            "  - the URL, credentials and stream path are correct\n"
            "  - the transport is supported (try rtsp_transport: udp)\n"
            "  - a firewall is not blocking RTSP (port 554 by default)"
        )

    # ------------------------------------------------------------------ #
    # FFmpeg configuration
    # ------------------------------------------------------------------ #

    def _ffmpeg_options(self) -> str:
        """Build the ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` value.

        Format is ``key;value`` pairs joined by ``|``.
        """
        timeout_us = int(max(0.0, self._config.open_timeout_sec) * 1_000_000)
        options = [
            f"rtsp_transport;{self._config.rtsp_transport}",
            # 'timeout' is the modern spelling, 'stimeout' the older one.
            # FFmpeg ignores options it does not recognise, so send both.
            f"timeout;{timeout_us}",
            f"stimeout;{timeout_us}",
            # A small probe keeps startup quick on a steady camera stream.
            "max_delay;500000",
        ]
        return "|".join(options)

    @contextmanager
    def _ffmpeg_environment(self):
        """Apply FFmpeg options for the duration of one capture construction.

        The variable is process-global and read only when a VideoCapture is
        built, so it is restored immediately afterwards rather than left to
        affect unrelated captures.
        """
        previous = os.environ.get(_FFMPEG_OPTIONS_ENV)
        os.environ[_FFMPEG_OPTIONS_ENV] = self._ffmpeg_options()
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_FFMPEG_OPTIONS_ENV, None)
            else:
                os.environ[_FFMPEG_OPTIONS_ENV] = previous

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _on_before_open(self, backend: CaptureBackend) -> None:
        if not self.url:
            raise SourceOpenError(
                "video.rtsp_url is not set. Set it in config.yaml, pass "
                "--rtsp-url, or set RTSP_URL."
            )
        if not self.url.lower().startswith(("rtsp://", "rtsps://")):
            raise SourceOpenError(
                f"video.rtsp_url must start with rtsp:// or rtsps:// - got {self.safe_url}"
            )

    def open(self):
        with self._ffmpeg_environment():
            info = super().open()

        self._reconnects = 0
        self._start_reader()
        logger.info("RTSP stream live: %s", self.safe_url)
        return info

    def _capture_params(self) -> list[int]:
        """Timeouts must be passed at construction, not set afterwards.

        Measured: setting ``CAP_PROP_OPEN_TIMEOUT_MSEC`` after the capture is
        built is silently ignored, and an unreachable camera then blocks for
        FFmpeg's 30 s default regardless of configuration. Passed here, an
        unreachable host fails in exactly ``open_timeout_sec``.
        """
        params: list[int] = []

        open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        if open_prop is not None and self._config.open_timeout_sec > 0:
            params += [int(open_prop), int(self._config.open_timeout_sec * 1000)]

        read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if read_prop is not None and self._config.read_timeout_sec > 0:
            params += [int(read_prop), int(self._config.read_timeout_sec * 1000)]

        return params

    def _configure_capture(self, capture: "cv2.VideoCapture") -> None:
        # Keep the driver-side buffer shallow; the reader thread does the real
        # staleness control, but this helps where the backend honours it.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _start_reader(self) -> None:
        if self._capture is None:
            return
        self._reader = _LatestFrameReader(self._capture, self._source_id)
        self._reader.start()

    def _stop_reader(self) -> None:
        if self._reader is not None:
            self._total_dropped += self._reader.frames_dropped
            self._reader.stop()
            self._reader = None

    def release(self) -> None:
        self._stop_reader()
        super().release()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def read(self) -> Optional[FramePacket]:
        """Return the newest frame, or ``None`` if none arrived in time.

        ``None`` is not an error: the pipeline keeps going and only gives up
        after ``processing.stall_timeout_sec`` without a frame.
        """
        if not self._is_open or self._reader is None:
            from app.sources.base import SourceNotOpenError

            raise SourceNotOpenError("RTSPSource is not open - call open() first")

        started = time.perf_counter()
        frame = self._reader.latest(self._config.read_timeout_sec)

        if frame is not None:
            self._reconnects = 0
            return FramePacket(
                frame=frame,
                frame_id=self._next_frame_id(),
                source_id=self._source_id,
                capture_ms=(time.perf_counter() - started) * 1000.0,
            )

        if self._reader.failed:
            self._reconnect()
            return None

        logger.debug("No RTSP frame within %.1fs", self._config.read_timeout_sec)
        return None

    # ------------------------------------------------------------------ #
    # Reconnection
    # ------------------------------------------------------------------ #

    def _reconnect(self) -> None:
        """Tear down and rebuild the capture, with exponential backoff."""
        if self._config.reconnect_attempts <= 0:
            raise FrameReadError(
                f"RTSP stream {self.safe_url} ended and reconnect is disabled "
                "(video.reconnect_attempts is 0)"
            )

        if self._reconnects >= self._config.reconnect_attempts:
            raise FrameReadError(
                f"RTSP stream {self.safe_url} failed after "
                f"{self._reconnects} reconnect attempt(s). The camera appears "
                "to be offline."
            )

        self._reconnects += 1
        self._total_reconnects += 1

        delay = min(
            self._config.reconnect_delay_sec * (2 ** (self._reconnects - 1)),
            self._config.reconnect_max_delay_sec,
        )
        logger.warning(
            "RTSP stream %s dropped - reconnecting in %.1fs (attempt %d/%d)",
            self.safe_url,
            delay,
            self._reconnects,
            self._config.reconnect_attempts,
        )

        self._stop_reader()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_open = False

        time.sleep(delay)

        try:
            with self._ffmpeg_environment():
                # Reopen without resetting the frame counter or reconnect count.
                OpenCVVideoSource.open(self)
            self._start_reader()
            logger.info("RTSP stream %s reconnected", self.safe_url)
        except SourceOpenError as exc:
            logger.warning("Reconnect failed: %s", exc)
            self._is_open = True  # stay "open" so the next read retries
            self._reader = _FailedReader()

    def stats_snapshot(self) -> dict:
        return {
            "source_id": self._source_id,
            "url": self.safe_url,
            "frames": self.frame_id,
            "reconnects": self._total_reconnects,
            "frames_dropped": self.frames_dropped,
        }


class _FailedReader:
    """Stand-in used between a failed reconnect and the next attempt."""

    failed = True
    frames_dropped = 0

    def latest(self, timeout: float) -> None:
        time.sleep(min(timeout, 0.1))
        return None

    def stop(self) -> None:
        return None
