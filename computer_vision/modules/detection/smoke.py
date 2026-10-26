"""Smoke Detection module.

One responsibility: report visual smoke candidates.

Kept separate from fire detection on purpose. Smoke usually precedes visible
flame, the two have different false-positive profiles (steam, fog, dust and
exhaust read as smoke), and a site may care about one and not the other.
Correlating the two is higher-level logic.

Requires a custom-trained model: no stock COCO model contains a smoke class.
"""

from __future__ import annotations

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.base import ModuleResult
from computer_vision.modules.detection.alerting import PersistenceAlert
from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_SMOKE_LABELS = ("smoke",)
EVENT_SMOKE = "smoke_detected"


class SmokeDetectionModule(LabelFilteredDetectionModule):
    """Detects visual smoke candidates.

    Config block::

        smoke_detection:
          enabled: true
          model: computer_vision/Model/custom/smoke.pt     # required
          labels: [smoke]
          confidence_threshold: 0.6
          min_frames: 10             # smoke_detected once seen in 10...
          window_frames: 15          # ...of the last 15 frames
          cooldown_seconds: 60
          severity: warning

    Emits ``smoke_detected``. Tune the defaults with the model on your own
    cameras: steam, fog, dust and exhaust are the usual false alarms.
    """

    module_name = "smoke_detection"
    version = "1.0.0"
    requires = ()

    default_labels = DEFAULT_SMOKE_LABELS
    requires_own_model = True
    model_hint = (
        "Steam, fog, dust and vehicle exhaust are common false positives; "
        "tune confidence_threshold against your own footage."
    )

    def initialize(self) -> None:
        self._alert = PersistenceAlert(
            self.module_name, EVENT_SMOKE, self.params,
            default_min_frames=10, default_window=15,
            default_cooldown=60.0, default_severity=Severity.WARNING,
        )
        super().initialize()

    def process(self, context: FrameContext) -> ModuleResult:
        result = super().process(context)
        event = self._alert.update(context, context.get_detections(self.module_name))
        if event is not None:
            context.add_event(event)
            self.log.warning(
                "%s (%s, %.2f)", event.event_type, event.subject["class"], event.data["confidence"]
            )
            result.data["events"] = [event.to_dict()]
        result.data["event_total"] = self._alert.total
        return result

    def cleanup(self) -> None:
        if hasattr(self, "_alert"):
            self._alert.reset()
        super().cleanup()
