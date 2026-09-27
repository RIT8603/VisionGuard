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

from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_FIRE_LABELS = ("fire",)


class FireDetectionModule(LabelFilteredDetectionModule):
    """Detects visual fire candidates.

    Config block::

        fire_detection:
          enabled: true
          model: models/fire.pt      # required
          labels: [fire]
          confidence_threshold: 0.6  # higher than usual; see the note above
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
