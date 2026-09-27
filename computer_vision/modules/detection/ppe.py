"""PPE Detection module.

One responsibility: detect personal protective equipment in the frame.

It reports what it sees - "a helmet is present here". It does **not** decide
whether anyone is in violation. Pairing PPE with the person wearing it, and
judging that against a site rule, is higher-level logic that consumes this
module's output (see Module_spec.md, "PPE Violation").

Requires a custom-trained model: no stock COCO model contains helmet, vest,
gloves or safety-boot classes.
"""

from __future__ import annotations

from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_PPE_LABELS = ("helmet", "vest", "gloves", "boots")


class PPEDetectionModule(LabelFilteredDetectionModule):
    """Detects PPE items.

    Config block::

        ppe_detection:
          enabled: true
          model: computer_vision/Model/custom/ppe.pt          # required - see below
          labels: [helmet, vest, gloves, boots]
          confidence_threshold: 0.5

    The ``labels`` must match the class names your model was trained with.
    Public PPE datasets disagree on naming ("helmet" vs "hardhat", "vest" vs
    "safety-vest"), so check your model's class list and set ``labels``
    accordingly.
    """

    module_name = "ppe_detection"
    version = "1.0.0"
    requires = ()

    default_labels = DEFAULT_PPE_LABELS
    requires_own_model = True
    model_hint = (
        "PPE datasets vary in class naming (helmet/hardhat, vest/safety-vest); "
        "set `labels` to match your model's class list."
    )
