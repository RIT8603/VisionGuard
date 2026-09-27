"""Smoke Detection module.

One responsibility: report visual smoke candidates.

Kept separate from fire detection on purpose. Smoke usually precedes visible
flame, the two have different false-positive profiles (steam, fog, dust and
exhaust read as smoke), and a site may care about one and not the other.
Correlating the two is higher-level logic.

Requires a custom-trained model: no stock COCO model contains a smoke class.
"""

from __future__ import annotations

from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_SMOKE_LABELS = ("smoke",)


class SmokeDetectionModule(LabelFilteredDetectionModule):
    """Detects visual smoke candidates.

    Config block::

        smoke_detection:
          enabled: true
          model: computer_vision/Model/custom/smoke.pt     # required
          labels: [smoke]
          confidence_threshold: 0.6
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
