"""Vehicle Detection module.

One responsibility: detect vehicles.

The default class list is the vehicle subset of COCO, so this works with the
stock model. Specialised vehicles such as forklifts are not COCO classes and
need a custom-trained model plus a ``labels`` override.
"""

from __future__ import annotations

from app.modules.detection.base import DEFAULT_COCO_MODEL, LabelFilteredDetectionModule

#: COCO vehicle classes. Module_spec also lists "forklift", which no stock
#: model contains - supply a custom model and set `labels` for that.
DEFAULT_VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle", "bicycle")


class VehicleDetectionModule(LabelFilteredDetectionModule):
    """Detects configured vehicle classes.

    Config block::

        vehicle_detection:
          enabled: true
          model: yolo11n.pt
          confidence_threshold: 0.4
          labels: [car, truck, bus, motorcycle, bicycle]
    """

    module_name = "vehicle_detection"
    version = "1.0.0"
    requires = ()

    default_labels = DEFAULT_VEHICLE_LABELS
    default_model = DEFAULT_COCO_MODEL
