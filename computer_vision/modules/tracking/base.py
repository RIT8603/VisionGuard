"""Shared machinery for tracking modules.

Every tracker in VisionGuard does the same thing: read a detector's output
from the shared context, hand it to an association algorithm, and publish the
resulting identities under its own name. What differs is *which* detector it
consumes - configuration, not code.

A tracker never runs a detector. That separation is a hard rule (Agent.md
§11): detection answers "is there one?", tracking answers "is this the same
one as last frame?".

As with the detection base, this is not a merging of modules. Each subclass
keeps its own ``module_name``, config block, enable switch, tracker instance
and ``FrameContext`` namespace - so ``person_tracking`` ids and
``vehicle_tracking`` ids are entirely independent.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from computer_vision.core.context import FrameContext
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType
from computer_vision.modules.tracking.tracker_core import IoUTracker

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_AGE = 30
DEFAULT_MIN_HITS = 2

#: Distance matching is used only while processed frames are at least this far
#: apart (about 6 fps or slower). Above that rate overlap alone is reliable,
#: and matching by closeness only adds chances to confuse two people.
SLOW_FRAME_GAP_SEC = 0.16
#: Weight of the newest frame gap in the running average.
_GAP_SMOOTHING = 0.2


class BaseTrackingModule(BaseAIModule):
    """Assigns persistent ids to one detector's output.

    Subclasses declare ``module_name``, ``source_module`` and a matching
    ``requires``. Config keys are common to all of them::

        <module_name>:
          enabled: true
          iou_threshold: 0.3
          max_age: 30
          min_hits: 2
          publish_coasting: false
          match_across_labels: false     # true: car/truck flicker keeps its id
          candidate_iou_threshold: 0.5   # default: iou_threshold
          max_center_distance: 0.8       # 0 = off; keeps ids at a low frame rate
                                         # (applied only below ~6 fps)
          compact_ids: true              # number only confirmed tracks

    Every track carries ``extra["travel"]``: how far it has moved from where
    it was first seen, in units of its own size. Analysis modules use it to
    tell something that walked in from a static false detection.

    When the detector publishes low-confidence candidates
    (``candidate_threshold`` on the detection module), they are used to keep
    an already-confirmed track going through a weak frame. Without them this
    module behaves exactly as before.
    """

    module_type: ClassVar[ModuleType] = ModuleType.TRACKING

    #: The detection module whose output this tracker consumes. Must also be
    #: named in ``requires`` so the registry orders and validates it.
    source_module: ClassVar[str] = ""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        if not self.source_module:
            raise ValueError(f"{type(self).__name__} must define 'source_module'")
        self._tracker: IoUTracker | None = None
        self._max_center_distance = 0.0
        self._last_frame_ts: float | None = None
        self._frame_gap: float | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Build the tracker. No model, no weights, no device."""
        self._tracker = IoUTracker(
            iou_threshold=float(self.params.get("iou_threshold", DEFAULT_IOU_THRESHOLD)),
            max_age=int(self.params.get("max_age", DEFAULT_MAX_AGE)),
            min_hits=int(self.params.get("min_hits", DEFAULT_MIN_HITS)),
            publish_coasting=bool(self.params.get("publish_coasting", False)),
            match_across_labels=bool(self.params.get("match_across_labels", False)),
            candidate_iou_threshold=(
                float(self.params["candidate_iou_threshold"])
                if self.params.get("candidate_iou_threshold") is not None
                else None
            ),
            max_center_distance=float(self.params.get("max_center_distance", 0.0)),
            compact_ids=bool(self.params.get("compact_ids", False)),
        )
        self._max_center_distance = self._tracker.max_center_distance
        self._last_frame_ts = None
        self._frame_gap = None
        self.log.info(
            "Ready: consuming '%s' (iou>=%.2f, max_age=%d, min_hits=%d)",
            self.source_module,
            self._tracker.iou_threshold,
            self._tracker.max_age,
            self._tracker.min_hits,
        )

    def cleanup(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
            self._tracker = None
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        if self._tracker is None:
            raise RuntimeError(
                f"{type(self).__name__}.process() called before initialize()"
            )

        detections = context.get_detections(self.source_module)
        candidates = context.get_candidates(self.source_module)
        self._adapt_to_frame_rate(context.monotonic_ts)
        track_states = self._tracker.update(
            detections, candidates, timestamp=context.monotonic_ts
        )

        timed = self._tracker.timed
        tracks = [state.as_track(self.module_name, timed) for state in track_states]
        context.add_tracks(self.module_name, tracks)

        return self.make_result(
            context,
            tracks=[track.to_dict() for track in tracks],
            data=self.summarise(tracks),
        )

    def _adapt_to_frame_rate(self, now: float) -> None:
        """Use distance matching only while frames are far apart in time."""
        assert self._tracker is not None
        if self._max_center_distance <= 0:
            return
        if self._last_frame_ts is not None:
            gap = max(0.0, now - self._last_frame_ts)
            self._frame_gap = (
                gap
                if self._frame_gap is None
                else _GAP_SMOOTHING * gap + (1.0 - _GAP_SMOOTHING) * self._frame_gap
            )
        self._last_frame_ts = now
        slow = self._frame_gap is not None and self._frame_gap >= SLOW_FRAME_GAP_SEC
        self._tracker.max_center_distance = self._max_center_distance if slow else 0.0

    def summarise(self, tracks) -> Dict[str, Any]:
        """Summary placed in the result's ``data``."""
        assert self._tracker is not None
        by_class: Dict[str, int] = {}
        for track in tracks:
            by_class[track.label] = by_class.get(track.label, 0) + 1

        return {
            "count": len(tracks),
            "by_class": by_class,
            "active": len(self._tracker.active_tracks),
            "total_ids": self._tracker.total_ids_issued,
            # Tells the output layer these detections are already represented
            # as tracks, so boxes are not drawn twice.
            "consumed_from": self.source_module,
        }

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def active_track_ids(self) -> List[int]:
        if self._tracker is None:
            return []
        return [state.track_id for state in self._tracker.active_tracks]
