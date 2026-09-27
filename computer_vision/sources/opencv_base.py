"""OpenCV implementation of the video-source contract.

Shared plumbing for every ``cv2.VideoCapture``-backed source. Subclasses only
declare *what* to open (:meth:`_capture_target`) and optionally *how* to
configure it (:meth:`_configure_capture`).
"""

from __future__ import annotations

import time
from typing import List, Optional, Union

import cv2

from computer_vision.core.config import CaptureBackend, VideoConfig
from computer_vision.sources.base import (
    BaseVideoSource,
    FramePacket,
    FrameReadError,
    SourceInfo,
    SourceNotOpenError,
    SourceOpenError,
)
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

# Numeric fallbacks keep this working on builds where a backend constant is
# compiled out (e.g. no DirectShow on Linux).
_BACKEND_IDS = {
    CaptureBackend.ANY: getattr(cv2, "CAP_ANY", 0),
    CaptureBackend.DSHOW: getattr(cv2, "CAP_DSHOW", 700),
    CaptureBackend.MSMF: getattr(cv2, "CAP_MSMF", 1400),
    CaptureBackend.V4L2: getattr(cv2, "CAP_V4L2", 200),
    CaptureBackend.FFMPEG: getattr(cv2, "CAP_FFMPEG", 1900),
    CaptureBackend.GSTREAMER: getattr(cv2, "CAP_GSTREAMER", 1800),
}

CaptureTarget = Union[int, str]


def fourcc_code(code: str) -> int:
    """Encode a four-character pixel format (``"MJPG"``) as an OpenCV FOURCC int.

    OpenCV 5 exposes ``cv2.VideoWriter.fourcc``; the module-level
    ``cv2.VideoWriter_fourcc`` is the older spelling. Both exist in 5.0, but
    only one is guaranteed on any given build, so try the modern name first.
    """
    encoder = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return int(encoder(*code))


class OpenCVVideoSource(BaseVideoSource):
    """Base class for OpenCV-backed sources."""

    source_type = "opencv"

    def __init__(self, source_id: str, config: VideoConfig) -> None:
        super().__init__(source_id)
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._info: Optional[SourceInfo] = None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def _capture_target(self) -> CaptureTarget:
        """The device index or URL/path handed to ``cv2.VideoCapture``."""
        raise NotImplementedError

    def _resolve_backend(self) -> CaptureBackend:
        """Backend to request. Overridden per source type for sensible defaults."""
        if self._config.backend is not CaptureBackend.AUTO:
            return self._config.backend
        return CaptureBackend.ANY

    def _configure_capture(self, capture: "cv2.VideoCapture") -> None:
        """Apply capture properties after a successful open. Optional."""

    def _capture_params(self) -> List[int]:
        """Flat ``[prop, value, ...]`` list applied *while* the capture opens.

        Some properties - notably the FFmpeg open timeout - only take effect
        when passed at construction. Setting them afterwards is too late,
        because the open they are meant to bound has already happened.
        """
        return []

    def _describe(self) -> str:
        return f"{type(self).__name__} -> {self._loggable_target()}"

    def _loggable_target(self) -> str:
        """The capture target in a form that is safe to write to a log.

        Overridden by sources whose target carries secrets - an RTSP URL
        embeds a real password that must never reach a log file.
        """
        return repr(self._capture_target())

    def _on_before_open(self, backend: CaptureBackend) -> None:
        """Hook for warning the user about a slow or unusual open. Optional."""

    def _handle_read_failure(self, attempt: int) -> bool:
        """Return True to retry the read, False to give up.

        The default policy sleeps briefly and retries up to
        ``video.read_retries`` times. RTSP will override this in Phase 5 to
        trigger a reconnect instead.
        """
        if attempt >= self._config.read_retries:
            return False
        time.sleep(self._config.read_retry_delay_sec)
        return True

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def open(self) -> SourceInfo:
        if self._is_open and self._capture is not None:
            return self.info

        target = self._capture_target()
        backend = self._resolve_backend()
        backend_id = _BACKEND_IDS.get(backend, _BACKEND_IDS[CaptureBackend.ANY])

        logger.info(
            "Opening %s (target=%s, backend=%s)",
            type(self).__name__,
            self._loggable_target(),
            backend.value,
        )
        self._on_before_open(backend)

        params = self._capture_params()
        started = time.perf_counter()
        try:
            if params:
                capture = cv2.VideoCapture(target, backend_id, params)
            else:
                capture = cv2.VideoCapture(target, backend_id)
        except cv2.error as exc:  # pragma: no cover - driver specific
            raise SourceOpenError(
                f"OpenCV failed to open {self._loggable_target()}: {exc}"
            ) from exc

        if not capture.isOpened():
            capture.release()
            raise SourceOpenError(self._open_failure_message(target, backend))

        self._capture = capture
        self._configure_capture(capture)

        self._info = self._read_info(capture, backend)
        self._is_open = True

        logger.info(
            "Source ready in %.0f ms: %s @ %dx%d, %.1f fps",
            (time.perf_counter() - started) * 1000.0,
            self._info.source_id,
            self._info.width,
            self._info.height,
            self._info.fps,
        )
        return self._info

    def read(self) -> Optional[FramePacket]:
        if not self._is_open or self._capture is None:
            raise SourceNotOpenError(
                f"{type(self).__name__} is not open - call open() first"
            )

        attempt = 0
        while True:
            started = time.perf_counter()
            ok, frame = self._capture.read()
            if ok and frame is not None:
                return FramePacket(
                    frame=frame,
                    frame_id=self._next_frame_id(),
                    source_id=self._source_id,
                    capture_ms=(time.perf_counter() - started) * 1000.0,
                )

            if not self._handle_read_failure(attempt):
                raise FrameReadError(
                    f"{type(self).__name__} failed to read a frame after "
                    f"{attempt + 1} attempt(s) from {self._capture_target()!r}"
                )
            attempt += 1

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            logger.info("Released source %s", self._source_id)
        self._capture = None
        self._is_open = False

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            raise SourceNotOpenError(
                f"{type(self).__name__} has no info - call open() first"
            )
        return self._info

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _read_info(self, capture: "cv2.VideoCapture", backend: CaptureBackend) -> SourceInfo:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return SourceInfo(
            source_id=self._source_id,
            source_type=self.source_type,
            width=width,
            height=height,
            fps=fps,
            backend=backend.value,
            description=self._describe(),
        )

    def _open_failure_message(self, target: CaptureTarget, backend: CaptureBackend) -> str:
        return (
            f"Could not open video source {target!r} using backend "
            f"'{backend.value}'."
        )
