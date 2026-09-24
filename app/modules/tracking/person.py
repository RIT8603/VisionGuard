"""Person Tracking module.

One responsibility: give each detected person a persistent id across frames.

It runs no detector. It reads the detections that ``person_detection`` already
published to the :class:`~app.core.context.FrameContext` and writes tracks
back. That separation is a hard rule (Agent.md §11): detection answers "is
there a person?", tracking answers "is this the same person as last frame?".
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from app.modules.tracking.base import (  # noqa: F401  (re-exported defaults)
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    BaseTrackingModule,
)


class PersonTrackingModule(BaseTrackingModule):
    """Assigns persistent track ids to person detections.

    Config block (``modules.person_tracking`` in config.yaml)::

        person_tracking:
          enabled: true
          iou_threshold: 0.3
          max_age: 30
          min_hits: 2
          publish_coasting: false

    ``requires`` makes the registry run ``person_detection`` first; enabling
    this module without it is rejected at startup with an actionable message.
    """

    module_name = "person_tracking"
    version = "1.1.0"
    requires = ("person_detection",)

    source_module = "person_detection"

    def summarise(self, tracks: Sequence[Any]) -> Dict[str, Any]:
        """People are all one class, so the per-class breakdown adds nothing."""
        summary = super().summarise(tracks)
        summary.pop("by_class", None)
        return summary
