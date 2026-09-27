"""The output sink that feeds the dashboard.

Lives in the backend, not in ``computer_vision``, because it is the bridge
between the two and the dependency must point one way: **backend knows about
the vision engine; the vision engine knows nothing about the backend.**

Keeping it here means the pipeline still runs with no server, no API and no
browser - it is just another :class:`BaseOutputSink` as far as the core is
concerned.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from backend.services.state import DashboardState
from computer_vision.core.config import ProcessingConfig
from computer_vision.core.context import FrameContext
from computer_vision.core.output import BaseOutputSink, untracked_detections
from computer_vision.utils.annotate import draw_detections, draw_overlays, draw_tracks
from computer_vision.utils.timing import PipelineStats


class WebSink(BaseOutputSink):
    """Publishes annotated frames and results to the dashboard.

    Encoding every frame to JPEG costs real time, so the stream is throttled
    independently of the pipeline: analysis still runs at full rate, only the
    *view* is capped. Results and events are published every frame regardless,
    because dropping an event to save bandwidth would be a bug.
    """

    def __init__(
        self,
        state: DashboardState,
        config: ProcessingConfig,
        stream_fps: float = 15.0,
        jpeg_quality: int = 75,
    ) -> None:
        self._state = state
        self._config = config
        self._min_interval = 1.0 / stream_fps if stream_fps > 0 else 0.0
        self._quality = int(jpeg_quality)
        self._last_encoded_at = 0.0

    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        self._state.publish(
            jpeg=self._encode(context),
            summary=context.summary(),
            stats=stats.as_dict(),
            events=[event.to_dict() for event in context.events],
        )
        return True

    def _encode(self, context: FrameContext) -> Optional[bytes]:
        now = time.perf_counter()
        if self._min_interval and (now - self._last_encoded_at) < self._min_interval:
            return None
        self._last_encoded_at = now

        frame = context.frame.copy()
        if context.overlays:
            draw_overlays(frame, context.overlays)
        if self._config.show_boxes:
            draw_detections(frame, untracked_detections(context))
            draw_tracks(frame, context.get_tracks())

        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        return buffer.tobytes() if ok else None

    def close(self) -> None:
        self._state.stop()
