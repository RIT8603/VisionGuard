"""Vehicle Tracking module.

One responsibility: give each detected vehicle a persistent id across frames.

Kept separate from object tracking on purpose. A site may track vehicles and
nothing else, vehicles tend to need different association settings from people
(they move faster, so a larger ``max_age`` and a lower ``iou_threshold`` often
help), and keeping the id sequences apart means vehicle ids are stable even
when unrelated objects come and go.
"""

from __future__ import annotations

from computer_vision.modules.tracking.base import BaseTrackingModule


class VehicleTrackingModule(BaseTrackingModule):
    """Assigns persistent track ids to vehicle detections.

    Config block::

        vehicle_tracking:
          enabled: true
          iou_threshold: 0.25   # vehicles move further between frames
          max_age: 45
          min_hits: 2

    Requires ``vehicle_detection``.
    """

    module_name = "vehicle_tracking"
    version = "1.0.0"
    requires = ("vehicle_detection",)

    source_module = "vehicle_detection"
