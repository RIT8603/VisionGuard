"""Local webcam / USB camera source — the development input.

Status: implemented (Phase 1).
"""

from __future__ import annotations

import sys

import cv2

from computer_vision.core.config import CaptureBackend, VideoConfig
from computer_vision.sources.base import SourceOpenError
from computer_vision.sources.opencv_base import CaptureTarget, OpenCVVideoSource, fourcc_code
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)


class WebcamSource(OpenCVVideoSource):
    """Captures from a locally attached camera via ``cv2.VideoCapture(device_id)``.

    Notes:
        * On Windows the default backend is MSMF. It costs a few seconds to
          open a camera, but measured on real hardware it sustains the camera's
          rated frame rate where DirectShow can get stuck on an uncompressed
          format at ~10 fps. Startup is paid once; frame rate is paid forever.
          Set ``video.backend: dshow`` for fast startup, or if a particular
          camera misbehaves under MSMF.
        * Requested width/height/fps are hints. Drivers commonly snap to the
          nearest supported mode, so the granted values are reported in
          :attr:`info` rather than assumed.
    """

    source_type = "webcam"

    def __init__(self, config: VideoConfig, source_id: str | None = None) -> None:
        super().__init__(source_id or config.resolved_source_id(), config)
        #: Set when the automatic backend failed and another one is being tried.
        self._backend_override: CaptureBackend | None = None

    def _capture_target(self) -> CaptureTarget:
        return self._config.device_id

    def open(self):
        """Open the device; on Windows, fall back from MSMF to DirectShow.

        Many HDMI/USB capture cards (and some webcams) only show up, or only
        deliver frames, under DirectShow. When the backend was left on
        ``auto`` that should not be the operator's problem to discover.
        """
        try:
            return super().open()
        except SourceOpenError:
            if not (
                sys.platform == "win32"
                and self._config.backend is CaptureBackend.AUTO
                and self._backend_override is None
            ):
                raise
            logger.warning(
                "Device %s did not open with MSMF; trying DirectShow (common for "
                "HDMI capture cards)",
                self._config.device_id,
            )
            self._backend_override = CaptureBackend.DSHOW
            try:
                return super().open()
            except SourceOpenError:
                self._backend_override = None
                raise

    def _resolve_backend(self) -> CaptureBackend:
        if self._backend_override is not None:
            return self._backend_override
        if self._config.backend is not CaptureBackend.AUTO:
            return self._config.backend
        if sys.platform == "win32":
            return CaptureBackend.MSMF
        if sys.platform.startswith("linux"):
            return CaptureBackend.V4L2
        return CaptureBackend.ANY

    def _on_before_open(self, backend: CaptureBackend) -> None:
        if backend is CaptureBackend.MSMF:
            logger.info(
                "MSMF can take several seconds to open a camera - please wait. "
                "Set video.backend: dshow for a faster start (often at a lower "
                "frame rate)."
            )

    def _configure_capture(self, capture: "cv2.VideoCapture") -> None:
        cfg = self._config

        # FOURCC must be set before the resolution: the pixel format determines
        # which modes the camera will even offer. Leaving a USB webcam on its
        # uncompressed default typically caps 720p at ~10 fps.
        if cfg.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, fourcc_code(cfg.fourcc))

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        # A shallow buffer keeps live latency low: we always want the newest
        # frame, not a backlog of stale ones.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        granted_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        granted_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if (granted_w, granted_h) != (cfg.width, cfg.height):
            logger.warning(
                "Camera %s did not grant %dx%d; using %dx%d instead",
                cfg.device_id,
                cfg.width,
                cfg.height,
                granted_w,
                granted_h,
            )

    def _describe(self) -> str:
        return f"Webcam device {self._config.device_id}"

    def _open_failure_message(self, target: CaptureTarget, backend: CaptureBackend) -> str:
        return (
            f"Could not open webcam device {target} (backend '{backend.value}').\n"
            "Check that:\n"
            "  - a camera is connected and not in use by another application\n"
            "  - the device id is correct (try 0, 1, 2 ... - an HDMI capture card\n"
            "    is usually 1 when the laptop has its own webcam)\n"
            "  - an HDMI source is plugged in and playing (some cards give no\n"
            "    video until they see a signal)\n"
            "  - camera access is permitted for this application\n"
            "    (Windows: Settings > Privacy & security > Camera)"
        )
