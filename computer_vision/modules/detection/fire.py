"""Fire Detection module.

One responsibility: report visual fire candidates.

"Candidate" is deliberate. Visual fire detection has a meaningful false-positive
rate - sunsets, brake lights, high-vis clothing and reflections all read as
fire to a detector. This module publishes detections with confidences; deciding
that an alarm is warranted (persistence over time, zone, corroboration with
smoke) is higher-level logic, not a job for the detector.

Requires a custom-trained model: no stock COCO model contains a fire class.
"""

from __future__ import annotations

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.base import ModuleResult
from computer_vision.modules.detection.alerting import PersistenceAlert
from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_FIRE_LABELS = ("fire",)
EVENT_FIRE = "fire_detected"


class FireDetectionModule(LabelFilteredDetectionModule):
    """Detects visual fire candidates.

    Config block::

        fire_detection:
          enabled: true
          model: computer_vision/Model/custom/fire.pt      # required
          labels: [fire]
          confidence_threshold: 0.6  # higher than usual; see the note above
          min_frames: 10             # fire_detected once seen in 10...
          window_frames: 15          # ...of the last 15 frames
          cooldown_seconds: 60
          severity: critical

    Emits ``fire_detected``. The defaults are a starting point: tune them with
    the model on your own cameras (brake lights, sunsets and hi-vis clothing
    are the usual false alarms).
    """

    module_name = "fire_detection"
    version = "1.0.0"
    requires = ()

    default_labels = DEFAULT_FIRE_LABELS
    requires_own_model = True
    model_hint = (
        "Visual fire detection is prone to false positives; a higher "
        "confidence_threshold and temporal confirmation upstream are advisable."
    )

    def initialize(self) -> None:
        self._alert = PersistenceAlert(
            self.module_name, EVENT_FIRE, self.params,
            default_min_frames=10, default_window=15,
            default_cooldown=60.0, default_severity=Severity.CRITICAL,
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
