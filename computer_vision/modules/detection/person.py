"""Person Detection module.

One responsibility: find people in the current frame and publish bounding
boxes with confidences.

It does not identify anyone, does not track, does not raise alerts and does not
touch a database. Those belong to other modules and to higher-level features
(see Module_spec.md, "Module Independence Rule").
"""

from __future__ import annotations

from computer_vision.modules.detection.base import (
    COCO_CLASS_IDS,
    DEFAULT_BACKEND,
    DEFAULT_COCO_MODEL,
    LabelFilteredDetectionModule,
)

#: COCO class id for "person". Restricting inference to it is a speed win.
COCO_PERSON_CLASS_ID = COCO_CLASS_IDS["person"]

DEFAULT_LABEL = "person"
DEFAULT_CONFIDENCE = 0.40

__all__ = [
    "COCO_PERSON_CLASS_ID",
    "DEFAULT_BACKEND",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_LABEL",
    "PersonDetectionModule",
]


class PersonDetectionModule(LabelFilteredDetectionModule):
    """Detects people using a pluggable detector backend.

    Config block (``modules.person_detection`` in config.yaml)::

        person_detection:
          enabled: true
          backend: ultralytics
          model: yolo11n.pt
          device: cpu
          imgsz: 640
          confidence_threshold: 0.4
          min_box_area: 0

    Every key other than ``enabled`` is forwarded to the backend, so backend
    specific options need no change here.
    """

    module_name = "person_detection"
    version = "1.1.0"
    requires = ()

    default_labels = (DEFAULT_LABEL,)
    default_model = DEFAULT_COCO_MODEL

    def summarise(self, detections) -> dict:
        """People are all one class, so a plain count is the useful summary."""
        return {"count": len(detections)}
