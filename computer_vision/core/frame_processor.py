"""Frame preprocessing — raw capture to analysis-ready FrameContext.

One deliberate design decision lives here: the frame is resized **once**, and
that single resolution is used for both inference and display. Modules
therefore produce bounding boxes in the same coordinate space the annotator
draws in, and no rescaling step can ever drift out of sync.
"""

from __future__ import annotations

import cv2

from computer_vision.core.config import ProcessingConfig
from computer_vision.core.context import FrameContext
from computer_vision.sources.base import FramePacket
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Turns a :class:`FramePacket` into a :class:`FrameContext`."""

    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config
        self._logged_resize = False

    def build_context(self, packet: FramePacket) -> FrameContext:
        """Preprocess ``packet`` and wrap it in a fresh context."""
        frame = packet.frame
        original_size = (packet.width, packet.height)

        if self._config.mirror:
            frame = cv2.flip(frame, 1)

        captured = frame
        frame = self._resize(frame)

        context = FrameContext(
            frame=frame,
            original_frame=captured if captured is not frame else None,
            frame_id=packet.frame_id,
            source_id=packet.source_id,
            monotonic_ts=packet.monotonic_ts,
            wall_time=packet.wall_time,
        )
        context.metadata.update(
            {
                "original_width": original_size[0],
                "original_height": original_size[1],
                "capture_ms": round(packet.capture_ms, 2),
                "mirrored": self._config.mirror,
            }
        )
        return context

    def prepare(self, frame):
        """Mirror and resize a raw frame exactly as ``build_context`` does.

        For a live viewer that shows frames the pipeline has not analysed:
        boxes computed on an analysed frame only line up on a frame that went
        through the same geometry.
        """
        if self._config.mirror:
            frame = cv2.flip(frame, 1)
        return self._resize(frame)

    def _resize(self, frame):
        target_width = self._config.target_width
        if not target_width:
            return frame

        height, width = frame.shape[:2]
        if width == target_width:
            return frame

        target_height = max(1, int(round(height * (target_width / width))))
        interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR

        if not self._logged_resize:
            logger.info(
                "Processing resolution: %dx%d -> %dx%d",
                width,
                height,
                target_width,
                target_height,
            )
            self._logged_resize = True

        return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
