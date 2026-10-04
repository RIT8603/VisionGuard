"""Weapon Detection module.

One responsibility: detect weapon-like objects in the frame and raise an
event when one is seen for several frames in a row.

Out of the box it uses the same COCO model as the other detectors, which
knows three relevant classes: **knife**, **baseball bat** and **scissors**.
That model has never seen a firearm - COCO has no gun class - so detecting
guns needs a custom-trained model: set ``model`` and ``labels`` to match it,
exactly as for PPE or fire.

Small, partly hidden objects are the hardest thing a general detector does.
Expect misses (a knife held low, at distance) and false alarms (a phone, an
umbrella handle). This is a prompt for a person to look, not a verdict.
"""

from __future__ import annotations

from typing import Any, Dict

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.analysis.common import make_event, parse_severity, positive_number
from computer_vision.modules.base import ModuleResult
from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_WEAPON_LABELS = ("knife", "baseball bat", "scissors")
EVENT_WEAPON = "weapon_detected"


class WeaponDetectionModule(LabelFilteredDetectionModule):
    """Detects weapon-like objects and emits ``weapon_detected``.

    Config block::

        weapon_detection:
          enabled: true
          model: <same as the other detectors, or a custom weapon model>
          labels: [knife, baseball bat, scissors]
          confidence_threshold: 0.4
          min_frames: 3          # consecutive frames before the event
          cooldown_seconds: 30   # quiet period after an event
          severity: critical
    """

    module_name = "weapon_detection"
    version = "1.0.0"
    requires = ()

    default_labels = DEFAULT_WEAPON_LABELS

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._streak = 0
        self._last_event_at: float | None = None
        self._total = 0

    def initialize(self) -> None:
        self._min_frames = max(1, int(self.params.get("min_frames", 3)))
        self._cooldown = positive_number(self.params, "cooldown_seconds", 30.0, allow_zero=True)
        self._severity = parse_severity(self.params, Severity.CRITICAL)
        super().initialize()

    def process(self, context: FrameContext) -> ModuleResult:
        result = super().process(context)
        detections = context.get_detections(self.module_name)
        self._streak = self._streak + 1 if detections else 0

        now = context.monotonic_ts
        quiet = self._last_event_at is not None and now - self._last_event_at < self._cooldown
        if self._streak >= self._min_frames and not quiet:
            self._last_event_at = now
            self._total += 1
            best = max(detections, key=lambda det: det.confidence)
            event = make_event(
                self.module_name, EVENT_WEAPON, context, self._severity,
                subject={"class": best.label, "count": len(detections)},
                data={
                    "confidence": round(float(best.confidence), 3),
                    "classes": sorted({det.label for det in detections}),
                    "consecutive_frames": self._streak,
                },
            )
            context.add_event(event)
            self.log.warning("%s (%s, %.2f)", event.event_type, best.label, best.confidence)
            result.data["events"] = [event.to_dict()]

        result.data["event_total"] = self._total
        return result

    def cleanup(self) -> None:
        self._streak = 0
        super().cleanup()
