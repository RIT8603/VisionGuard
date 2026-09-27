"""Object Tracking module.

One responsibility: give each detected object a persistent id across frames.

Consumes ``object_detection``, which may report many classes at once. The
association algorithm never matches across labels, so a car cannot inherit a
dog's id even at perfect box overlap - ids are per-object, and the class rides
along with them.

Ids are drawn from a single sequence shared by every class this module tracks,
so a run may produce ``#1 person``, ``#2 car``, ``#3 person``. That is
deliberate: an id identifies a thing, not a thing-of-a-class.
"""

from __future__ import annotations

from computer_vision.modules.tracking.base import BaseTrackingModule


class ObjectTrackingModule(BaseTrackingModule):
    """Assigns persistent track ids to object detections.

    Config block::

        object_tracking:
          enabled: true
          iou_threshold: 0.3
          max_age: 30
          min_hits: 2

    Requires ``object_detection``. Narrow what gets tracked by narrowing that
    module's ``labels`` rather than by filtering here - tracking something the
    detector was told to ignore is impossible, and filtering twice is a place
    for the two configurations to drift apart.
    """

    module_name = "object_tracking"
    version = "1.0.0"
    requires = ("object_detection",)

    source_module = "object_detection"
