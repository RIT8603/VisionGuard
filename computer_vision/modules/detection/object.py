"""Object Detection module.

One responsibility: detect the object classes the configuration asks for.

With no ``labels`` set it reports everything the model knows (80 COCO classes
by default), which makes it the general-purpose detector. Narrow it with
``labels`` when only some classes matter - that is cheaper and quieter than
filtering downstream.
"""

from __future__ import annotations

from computer_vision.modules.detection.base import DEFAULT_COCO_MODEL, LabelFilteredDetectionModule


class ObjectDetectionModule(LabelFilteredDetectionModule):
    """Detects configured object classes.

    Config block::

        object_detection:
          enabled: true
          model: yolo11n.pt
          confidence_threshold: 0.4
          labels: []          # empty = every class the model knows

    Note it does not exclude people: ``person`` is a COCO class like any
    other. Enable ``person_detection`` as well when you want people tracked
    separately - each module keeps its own namespace in the FrameContext, so
    the two never collide.
    """

    module_name = "object_detection"
    version = "1.0.0"
    requires = ()

    #: Empty means "report every class the model produces".
    default_labels = ()
    default_model = DEFAULT_COCO_MODEL
