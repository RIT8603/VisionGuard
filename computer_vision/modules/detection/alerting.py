"""Turning per-frame detections into one event when something persists.

A detector answers "is there one in this frame?". An alarm needs more: a fire
detector fires on a single brake light or reflection, so an event is raised
only when the thing is seen in most of the recent frames, and then not again
for a cooldown. Fire and smoke detection share this; weapon detection keeps its
own consecutive-frames rule.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence

from computer_vision.core.context import Detection, Event, FrameContext, Severity
from computer_vision.modules.analysis.common import make_event, parse_severity, positive_number
from computer_vision.modules.base import ModuleInitializationError


class PersistenceAlert:
    """``event_type`` once a detection is present in ``min_frames`` of the last ``window_frames``.

    Config keys (read from the owning module's params)::

        min_frames: 10        # frames with a detection, out of...
        window_frames: 15     # ...this many recent frames
        cooldown_seconds: 60  # quiet period after an event
        severity: critical

    Tolerating a few missing frames inside the window matters: a real fire or
    smoke plume is not detected on every single frame, so "N in a row" would
    keep restarting.
    """

    def __init__(
        self,
        module_name: str,
        event_type: str,
        params: Dict[str, Any],
        *,
        default_min_frames: int,
        default_window: int,
        default_cooldown: float,
        default_severity: Severity,
    ) -> None:
        self.module_name = module_name
        self.event_type = event_type
        self.min_frames = max(1, int(params.get("min_frames", default_min_frames)))
        self.window = max(1, int(params.get("window_frames", max(default_window, self.min_frames))))
        if self.window < self.min_frames:
            raise ModuleInitializationError(
                f"{module_name}.window_frames ({self.window}) must be >= min_frames ({self.min_frames})"
            )
        self.cooldown = positive_number(params, "cooldown_seconds", default_cooldown, allow_zero=True)
        self.severity = parse_severity(params, default_severity)
        self._seen: Deque[bool] = deque(maxlen=self.window)
        self._last_event_at: Optional[float] = None
        self.total = 0

    def update(self, context: FrameContext, detections: Sequence[Detection]) -> Optional[Event]:
        """Record this frame; return the event when one is due."""
        self._seen.append(bool(detections))
        frames_seen = sum(self._seen)
        now = context.monotonic_ts
        quiet = self._last_event_at is not None and now - self._last_event_at < self.cooldown
        if not detections or frames_seen < self.min_frames or quiet:
            return None

        self._last_event_at = now
        self.total += 1
        best = max(detections, key=lambda det: det.confidence)
        width, height = max(context.width, 1), max(context.height, 1)
        return make_event(
            self.module_name, self.event_type, context, self.severity,
            subject={"class": best.label, "count": len(detections)},
            data={
                "confidence": round(float(best.confidence), 3),
                "classes": sorted({det.label for det in detections}),
                "frames_seen": frames_seen,
                "window_frames": self.window,
                # Where in the picture, as frame fractions (resolution-independent).
                "bbox": [
                    round(best.bbox.x1 / width, 4), round(best.bbox.y1 / height, 4),
                    round(best.bbox.x2 / width, 4), round(best.bbox.y2 / height, 4),
                ],
            },
        )

    def reset(self) -> None:
        self._seen.clear()
        self._last_event_at = None
