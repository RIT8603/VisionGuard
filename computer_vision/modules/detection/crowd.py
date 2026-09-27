"""Crowd Detection module.

One responsibility: turn person detections into crowd measurements.

It runs **no model**. Person detection already found the people; running a
second detector to count them would be duplicated inference for no gain
(Agent.md §9). This module consumes ``person_detection`` from the shared
context, exactly as ``person_tracking`` does.

It reports measurements and a threshold state. It does not raise alerts, write
incidents or notify anyone - that is higher-level logic consuming this output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from computer_vision.core.context import Detection, FrameContext
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType

DEFAULT_CROWD_THRESHOLD = 5


class CrowdDetectionModule(BaseAIModule):
    """Counts people and estimates how densely they occupy the frame.

    Config block::

        crowd_detection:
          enabled: true
          threshold: 5          # people at or above this = crowd
          source_module: person_detection

    Output ``data``::

        {
          "count": 7,
          "threshold": 5,
          "is_crowded": true,
          "occupancy": 0.23,        # fraction of frame area covered by people
          "mean_confidence": 0.81
        }
    """

    module_name = "crowd_detection"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("person_detection",)

    #: The detector whose output is counted.
    source_module = "person_detection"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._threshold = int(self.params.get("threshold", DEFAULT_CROWD_THRESHOLD))
        self._peak_count = 0

    def initialize(self) -> None:
        """Nothing to load. Counting is arithmetic, not inference."""
        if self._threshold < 1:
            raise ValueError("crowd_detection.threshold must be >= 1")
        self.log.info(
            "Ready: consuming '%s', crowd at >= %d people",
            self.source_module,
            self._threshold,
        )

    def process(self, context: FrameContext) -> ModuleResult:
        people = context.get_detections(self.source_module)
        count = len(people)
        self._peak_count = max(self._peak_count, count)

        return self.make_result(
            context,
            data={
                "count": count,
                "threshold": self._threshold,
                "is_crowded": count >= self._threshold,
                "occupancy": self._occupancy(people, context),
                "mean_confidence": self._mean_confidence(people),
                "peak_count": self._peak_count,
                # Crowd measurement consumes the detections but does not
                # replace them, so the boxes are still drawn by the detector.
            },
        )

    @property
    def peak_count(self) -> int:
        return self._peak_count

    @staticmethod
    def _occupancy(people: Sequence[Detection], context: FrameContext) -> float:
        """Fraction of the frame covered by person boxes.

        Overlapping boxes are counted once each, so this is a rough density
        proxy rather than true coverage - it is useful for comparing frames
        from the same camera, not as an absolute measure.
        """
        frame_area = float(context.width * context.height)
        if frame_area <= 0 or not people:
            return 0.0
        covered = sum(person.bbox.area for person in people)
        return round(min(1.0, covered / frame_area), 4)

    @staticmethod
    def _mean_confidence(people: Sequence[Detection]) -> float:
        if not people:
            return 0.0
        return round(sum(p.confidence for p in people) / len(people), 4)
