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
same :class:`~computer_vision.sources.base.FramePacket` a webcam does.
"""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from computer_vision.core.config import CaptureBackend, VideoConfig
from computer_vision.sources.base import FramePacket, FrameReadError, SourceOpenError
from computer_vision.sources.opencv_base import CaptureTarget, OpenCVVideoSource
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

#: OpenCV reads this at VideoCapture construction to configure FFmpeg.
_FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


#: Query parameter names that carry credentials on common NVR/IP cameras
#: (e.g. Dahua/Axis style ``?user=admin&password=...``). Matched as substrings.
_SECRET_QUERY_KEYS = ("pass", "pwd", "token", "secret", "auth", "key", "sig", "credential")


def _redact_query(query: str) -> str:
    if not query:
        return query
    parts = []
    for item in query.split("&"):
        name, sep, value = item.partition("=")
        if sep and value and any(k in name.lower() for k in _SECRET_QUERY_KEYS):
            item = f"{name}=***"
        parts.append(item)
    return "&".join(parts)


def redact_url(url: str) -> str:
    """Hide the secrets in an RTSP URL so it is safe to log.

    ``rtsp://admin:hunter2@10.0.0.5/stream`` -> ``rtsp://admin:***@10.0.0.5/stream``
    ``rtsp://cam/stream?user=admin&password=x`` -> ``...?user=admin&password=***``

    Credentials in camera URLs are routinely real and routinely reused, so no
    code path may ever write one to a log file or an error message. The user
    name is kept on purpose: it identifies which account is failing and is
    not a secret on its own.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        # Unparseable: fall back to a blunt regex rather than risk leaking.
        blunt = re.sub(r"//[^/@]*@", "//***@", url)
        return re.sub(
            r"([?&][^=&]*(?:pass|pwd|token|secret|auth|key|sig|credential)[^=&]*=)[^&#]*",
            r"\1***",
            blunt,
            flags=re.I,
        )

    query = _redact_query(parts.query)

    # A password can be present without a username (rtsp://:secret@host),
    # so both must be checked - testing only username leaks that form.
    if not parts.username and not parts.password:
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))

    userinfo = parts.username or ""
    if parts.password:
        userinfo += ":***"

    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal: keep its brackets
        host = f"[{host}]"
    if port:
        host = f"{host}:{port}"

    return urlunsplit(
        (parts.scheme, f"{userinfo}@{host}", parts.path, query, parts.fragment)
    )


class _LatestFrameReader:
    """Drains a capture on its own thread, keeping only the newest frame.

    This is the piece that keeps live latency bounded. The reader never blocks
    on the consumer; if the pipeline is slow, frames are discarded, and the
    count is reported so the drop rate is visible rather than silent.
    """

    def __init__(
        self, capture: "cv2.VideoCapture", name: str, join_timeout: float = 2.0
    ) -> None:
        self._capture = capture
        self._name = name
        self._join_timeout = max(0.1, float(join_timeout))
        self._condition = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        #: The newest frame again, but never cleared by latest(): what a live
        #: viewer shows. ``_sequence`` counts captured frames so the viewer
        #: can tell a new one from the one it already drew.
        self._newest: Optional[np.ndarray] = None
        self._sequence = 0
        self._stop = threading.Event()
        self._failed = False
        self._dropped = 0
        self._thread: Optional[threading.Thread] = None
        #: Set once _run() has returned and will never touch the capture again.
        self._exited = False
        #: Set when stop() could not join the thread: the thread then releases
        #: the capture itself, after its blocking read returns.
        self._release_on_exit = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"rtsp-reader-{self._name}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
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
                    self._newest = frame
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._exited = True
                release = self._release_on_exit
            if release:
                # stop() gave up waiting while this thread was inside read();
                # releasing from here guarantees release never races a read.
                try:
                    self._capture.release()
                except Exception:  # noqa: BLE001 - best effort on a dead stream
                    logger.debug("Deferred capture release failed", exc_info=True)

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

    def peek(self) -> tuple[Optional[np.ndarray], int]:
        """The newest frame and its sequence number, leaving it for latest()."""
        with self._condition:
            return self._newest, self._sequence

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failed and self._frame is None

    @property
    def frames_dropped(self) -> int:
        with self._condition:
            return self._dropped

    def stop(self) -> bool:
        """Stop the thread.

        Returns True when the caller may release the capture now. Returns
        False when the thread is still blocked inside ``capture.read()``;
        ownership of the capture then passes to the thread, which releases it
        as soon as that read returns. Releasing a ``VideoCapture`` while
        another thread is reading it is undefined behaviour in the native
        backend (crashes, not exceptions).
        """
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        self._thread = None
        if thread is None:
            return True
        if thread.is_alive():
            # Wait at least one full read timeout, so a normally-behaving
            # stream always finishes its read before we return.
            thread.join(timeout=self._join_timeout)
        with self._condition:
            if self._exited or not thread.is_alive():
                return True
            self._release_on_exit = True
        logger.warning(
            "RTSP reader for %s is still blocked in a read; it will release "
            "the stream when the read returns",
            self._name,
        )
        return False


class RTSPSource(OpenCVVideoSource):
    """Reads frames from an RTSP stream."""

    source_type = "rtsp"

    def __init__(self, config: VideoConfig, source_id: str | None = None) -> None:
        super().__init__(source_id or config.resolved_source_id(), config)
        self._reader: Optional[_LatestFrameReader] = None
        self._recovering = False
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
    def is_recovering(self) -> bool:
        """True from a dropped stream until the next frame (or a clean reopen)."""
        return self._recovering

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
        self._reader = _LatestFrameReader(
            self._capture,
            self._source_id,
            # Longer than one blocking read, so stop() normally joins cleanly.
            join_timeout=max(2.0, self._config.read_timeout_sec + 1.0),
        )
        self._reader.start()

    def _stop_reader(self) -> bool:
        """Stop the reader. Returns True if the capture may be released here."""
        if self._reader is None:
            return True
        self._total_dropped += self._reader.frames_dropped
        safe = self._reader.stop()
        self._reader = None
        return safe

    def _drop_capture(self) -> None:
        """Stop reading and release the capture without racing the reader."""
        if not self._stop_reader():
            # The reader thread now owns the capture and releases it.
            self._capture = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def release(self) -> None:
        if not self._stop_reader():
            self._capture = None  # released by the reader thread instead
        self._recovering = False
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
            from computer_vision.sources.base import SourceNotOpenError

            raise SourceNotOpenError("RTSPSource is not open - call open() first")

        started = time.perf_counter()
        frame = self._reader.latest(self._config.read_timeout_sec)

        if frame is not None:
            self._reconnects = 0
            self._recovering = False
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

    def peek(self) -> tuple[Optional[np.ndarray], int]:
        """Newest captured frame for a live viewer; see ``BaseVideoSource.peek``.

        The sequence number is offset per connection, so it keeps changing
        across a reconnect instead of starting again from 1.
        """
        reader = self._reader
        if not self._is_open or reader is None:
            return None, 0
        frame, sequence = reader.peek()
        if frame is None:
            return None, 0
        return frame, (self._total_reconnects << 32) | sequence

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
        self._recovering = True

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

        self._drop_capture()
        self._is_open = False

        time.sleep(delay)

        try:
            with self._ffmpeg_environment():
                # Reopen without resetting the frame counter or reconnect count.
                OpenCVVideoSource.open(self)
            self._start_reader()
            # Reopened. If the new stream then hangs without failing, the
            # pipeline's stall timeout applies again.
            self._recovering = False
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

    def peek(self) -> tuple[None, int]:
        return None, 0

    def stop(self) -> bool:
        return True
