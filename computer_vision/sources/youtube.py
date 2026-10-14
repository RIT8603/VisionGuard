"""YouTube Live stream and video source for VisionGuard.

Uses yt-dlp to resolve public YouTube live streams and videos to direct HLS (m3u8)
or MP4 stream manifests and feeds frames to cv2.VideoCapture using FFmpeg.

Includes an asynchronous background reader thread (_LatestFrameReader) to drop stale
buffered frames and keep live inference latency near zero.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from typing import Optional

import cv2
import numpy as np

from computer_vision.core.config import CaptureBackend, VideoConfig
from computer_vision.sources.base import (
    FramePacket,
    FrameReadError,
    SourceNotOpenError,
    SourceOpenError,
    is_valid_frame,
)
from computer_vision.sources.opencv_base import CaptureTarget, OpenCVVideoSource
from computer_vision.sources.rtsp import _FailedReader, _LatestFrameReader
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

_YOUTUBE_URL_REGEX = re.compile(
    r"^https?://(www\.|m\.)?(youtube\.com/(watch\?v=|live/|embed/|shorts/)|youtu\.be/)[a-zA-Z0-9_\-]+",
    re.IGNORECASE,
)


def is_youtube_url(url: str | None) -> bool:
    """True if ``url`` looks like a standard YouTube watch, live, or short link."""
    if not url:
        return False
    return bool(_YOUTUBE_URL_REGEX.match(url.strip()))


def _find_node_path() -> Optional[str]:
    """Locate Node.js runtime if present, to assist yt-dlp JavaScript extraction."""
    candidates = [
        r"C:\Users\user\AppData\Local\hermes\tools\node-26.7.0-win32-x64\node.exe",
        shutil.which("node"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


class YouTubeSource(OpenCVVideoSource):
    """Streams live video from a YouTube Live or video URL."""

    source_type = "youtube"

    def __init__(self, config: VideoConfig, source_id: str | None = None) -> None:
        super().__init__(source_id or config.resolved_source_id(), config)
        self._stream_url: Optional[str] = None
        self._reader: Optional[_LatestFrameReader] = None
        self._recovering = False
        self._reconnects = 0
        self._total_reconnects = 0
        self._total_dropped = 0
        self._stream_title: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Properties & Identity
    # ------------------------------------------------------------------ #

    @property
    def url(self) -> str:
        return self._config.youtube_url or ""

    @property
    def safe_url(self) -> str:
        return self.url

    @property
    def stream_title(self) -> Optional[str]:
        return self._stream_title

    @property
    def is_recovering(self) -> bool:
        return self._recovering

    @property
    def total_reconnects(self) -> int:
        return self._total_reconnects

    @property
    def frames_dropped(self) -> int:
        dropped = self._total_dropped
        if self._reader is not None:
            dropped += self._reader.frames_dropped
        return dropped

    def _capture_target(self) -> CaptureTarget:
        if not self._stream_url:
            self._resolve_stream_url()
        return self._stream_url or self.url

    def _resolve_backend(self) -> CaptureBackend:
        if self._config.backend is not CaptureBackend.AUTO:
            return self._config.backend
        return CaptureBackend.FFMPEG

    def _describe(self) -> str:
        title = f" ({self._stream_title})" if self._stream_title else ""
        return f"YouTube {self.safe_url}{title}"

    def _loggable_target(self) -> str:
        return self.safe_url

    def _open_failure_message(self, target: CaptureTarget, backend: CaptureBackend) -> str:
        return (
            f"Could not open YouTube stream {self.safe_url}.\n"
            "Check that the stream is currently live and accessible."
        )

    # ------------------------------------------------------------------ #
    # Stream Resolution via yt-dlp
    # ------------------------------------------------------------------ #

    def _resolve_stream_url(self) -> str:
        """Extract a playable direct stream URL using yt-dlp."""
        if not self.url:
            raise SourceOpenError(
                "video.youtube_url is not set. Provide a valid YouTube URL."
            )
        if not is_youtube_url(self.url):
            raise SourceOpenError(
                f"Invalid YouTube URL: {self.safe_url}. Must be a valid youtube.com or youtu.be link."
            )

        try:
            import yt_dlp
        except ImportError as exc:
            raise SourceOpenError(
                "yt-dlp is not installed. Run 'pip install yt-dlp' to support YouTube streams."
            ) from exc

        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
        }
        node_path = _find_node_path()
        if node_path:
            ydl_opts["js_runtimes"] = {"node": {"path": node_path}}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
        except Exception as exc:
            msg = str(exc)
            if "ERROR: [youtube]" in msg:
                msg = msg.split("ERROR: [youtube]", 1)[1].strip()
            raise SourceOpenError(f"Could not load YouTube stream ({msg})") from exc

        if not info:
            raise SourceOpenError(f"No stream information returned for {self.safe_url}")

        self._stream_title = info.get("title")

        # Pick best format (prioritize video streams <= 1080p for efficient inference)
        formats = info.get("formats") or []
        video_formats = [f for f in formats if f.get("url")]
        stream_url: Optional[str] = None
        if video_formats:
            # Prefer formats <= 1080p
            candidates = [f for f in video_formats if (f.get("height") or 0) <= 1080]
            best = max(candidates or video_formats, key=lambda f: f.get("height") or 0)
            stream_url = best.get("url")

        if not stream_url:
            stream_url = info.get("url")

        if not stream_url:
            raise SourceOpenError(
                f"No playable video format found for YouTube URL {self.safe_url}."
            )

        self._stream_url = stream_url
        logger.debug("Resolved YouTube stream URL for %s (title: %s)", self.safe_url, self._stream_title)
        return stream_url

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _on_before_open(self, backend: CaptureBackend) -> None:
        if not self._stream_url:
            self._resolve_stream_url()

    def _configure_capture(self, capture: "cv2.VideoCapture") -> None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _start_reader(self) -> None:
        if self._capture is None:
            return
        self._reader = _LatestFrameReader(
            self._capture,
            self._source_id,
            join_timeout=max(2.0, self._config.read_timeout_sec + 1.0),
        )
        self._reader.start()

    def _stop_reader(self) -> bool:
        """Stop background reader. Returns True if capture can be released immediately."""
        if self._reader is None:
            return True
        self._total_dropped += self._reader.frames_dropped
        safe = self._reader.stop()
        self._reader = None
        return safe

    def _drop_capture(self) -> None:
        """Stop reader and release capture cleanly."""
        if not self._stop_reader():
            self._capture = None
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def open(self):
        if not self._stream_url:
            self._resolve_stream_url()
        try:
            info = super().open()
        except SourceOpenError as exc:
            logger.error("Failed to open YouTube source %s: %s", self.safe_url, exc)
            raise

        self._reconnects = 0
        self._start_reader()
        logger.info("YouTube stream live: %s (%s)", self.safe_url, self._stream_title or "live")
        return info

    def release(self) -> None:
        if not self._stop_reader():
            self._capture = None
        self._recovering = False
        super().release()

    # ------------------------------------------------------------------ #
    # Reading Frames
    # ------------------------------------------------------------------ #

    def read(self) -> Optional[FramePacket]:
        if not self._is_open or self._reader is None:
            raise SourceNotOpenError("YouTubeSource is not open - call open() first")

        started = time.perf_counter()
        frame = self._reader.latest(self._config.read_timeout_sec)

        if frame is not None and is_valid_frame(frame):
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

        logger.debug("No YouTube frame within %.1fs", self._config.read_timeout_sec)
        return None

    def peek(self) -> tuple[Optional[np.ndarray], int]:
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
        """Tear down and reconnect with exponential backoff and refreshed URL tokens."""
        if self._config.reconnect_attempts <= 0:
            raise FrameReadError(
                f"YouTube stream {self.safe_url} ended and reconnect is disabled."
            )

        if self._reconnects >= self._config.reconnect_attempts:
            raise FrameReadError(
                f"YouTube stream {self.safe_url} failed after {self._reconnects} reconnect attempt(s). "
                "The stream appears to be offline or finished."
            )

        self._reconnects += 1
        self._total_reconnects += 1
        self._recovering = True

        delay = min(
            self._config.reconnect_delay_sec * (2 ** (self._reconnects - 1)),
            self._config.reconnect_max_delay_sec,
        )
        logger.warning(
            "YouTube stream stalled/dropped (%s). Reconnecting in %.1fs (attempt %d/%d)...",
            self.safe_url,
            delay,
            self._reconnects,
            self._config.reconnect_attempts,
        )
        self._drop_capture()
        self._is_open = False

        time.sleep(delay)

        # Re-resolve stream URL: tokens expire in long-lived live streams!
        try:
            self._resolve_stream_url()
            super().open()
            self._start_reader()
            self._recovering = False
            logger.info("YouTube stream reconnected successfully: %s", self.safe_url)
        except Exception as exc:
            logger.warning("YouTube reconnect attempt %d failed: %s", self._reconnects, exc)
            self._is_open = True
            self._reader = _FailedReader()
