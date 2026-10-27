"""Abandoned Object module.

One responsibility: report a bag-like object that has sat still, with nobody
near it, for longer than a configured time.

Three conditions must hold together, continuously:

* the object is one of the watched classes (backpack, handbag, suitcase),
* it has not moved (its centre stays within a tolerance of where it was), and
* no person is within reach of it.

A person walking past resets the clock - an owner standing next to their bag
is the normal case, not an incident. It runs no model: it reads
``object_tracking`` for the objects and ``person_detection`` for the people.

It is a heuristic. A bag whose owner sits two metres away looks the same to a
camera as one that was left behind; tune ``owner_distance`` for the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from computer_vision.core.context import BBox, FrameContext, Severity
from computer_vision.modules.analysis.common import (
    SubjectMemory,
    make_event,
    parse_severity,
    positive_number,
)
from computer_vision.modules.analysis.geometry import subject_key
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType

DEFAULT_LABELS = ("backpack", "handbag", "suitcase")
EVENT_ABANDONED = "abandoned_object"
EVENT_COLLECTED = "abandoned_object_removed"


@dataclass
class _Resting:
    label: str
    anchor: Tuple[float, float]
    size: float
    unattended_since: float
    reported: bool = False


def _distance_to_box(point: Tuple[float, float], box: BBox) -> float:
    """Shortest distance from a point to a box (0 when the point is inside)."""
    dx = max(box.x1 - point[0], 0.0, point[0] - box.x2)
    dy = max(box.y1 - point[1], 0.0, point[1] - box.y2)
    return math.hypot(dx, dy)


class AbandonedObjectModule(BaseAIModule):
    """Emits ``abandoned_object`` for an unattended, stationary bag.

    Config block::

        abandoned_object:
          enabled: true
          labels: [backpack, handbag, suitcase]
          seconds: 20            # still and unattended for this long
          owner_distance: 1.5    # "near": this many object-sizes from a person
          move_tolerance: 0.5    # "still": centre moves less than this many object-sizes
          severity: warning

    Requires ``object_tracking`` and ``person_detection``.
    """

    module_name = "abandoned_object"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("object_tracking", "person_detection")

    def initialize(self) -> None:
        self._labels = tuple(str(label) for label in (self.params.get("labels") or DEFAULT_LABELS))
        self._seconds = positive_number(self.params, "seconds", 20.0)
        self._owner_distance = positive_number(self.params, "owner_distance", 1.5)
        self._tolerance = positive_number(self.params, "move_tolerance", 0.5)
        self._severity = parse_severity(self.params, Severity.WARNING)
        # Long memory on purpose: a person walking in front of a bag hides it
        # from the detector, and the clock should survive that.
        self._merge_bag_labels = bool(self.params.get("merge_bag_labels", False))
        self._memory: SubjectMemory[_Resting] = SubjectMemory(
            forget_seconds=positive_number(self.params, "forget_seconds", 10.0)
        )
        self._total = 0
        self.log.info(
            "Ready: watching %s, abandoned after %.0fs unattended",
            list(self._labels), self._seconds,
        )

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        people = [
            det.bbox
            for det in context.get_detections("person_detection")
        ]
        events = []
        abandoned: List[int] = []

        for track in context.get_tracks("object_tracking"):
            if track.label not in self._labels:
                continue
            key = subject_key(track.source_module, track.track_id)
            centre = track.bbox.center
            size = max(track.bbox.width, track.bbox.height, 1.0)

            state = self._memory.get(key) or self._inherit(key, track.label, centre, size, now)
            moved = state is not None and math.dist(centre, state.anchor) > self._tolerance * state.size
            attended = any(
                _distance_to_box(centre, person) <= self._owner_distance * size for person in people
            )

            if state is None or moved or attended:
                if state is not None and state.reported and moved:
                    events.append(self._event(EVENT_COLLECTED, context, track, state, now, Severity.INFO))
                self._memory.put(
                    key, _Resting(label=track.label, anchor=centre, size=size, unattended_since=now), now
                )
                continue

            self._memory.touch(key, now)
            if now - state.unattended_since < self._seconds:
                continue

            abandoned.append(track.track_id)
            if not state.reported:
                state.reported = True
                events.append(self._event(EVENT_ABANDONED, context, track, state, now, self._severity))

        self._memory.expire(now)
        for event in events:
            context.add_event(event)
            self.log.info("%s", event.describe())
        self._total += sum(1 for event in events if event.event_type == EVENT_ABANDONED)

        return self.make_result(
            context,
            data={
                "watching": len(self._memory),
                "abandoned": abandoned,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def _inherit(self, key, label: str, centre, size: float, now: float) -> _Resting | None:
        """Carry the clock over when the tracker gives the same bag a new id.

        A passer-by occluding a bag ends its track; when it reappears it is a
        new track in the same spot. Without this the timer would restart every
        time someone walked by, and a busy corridor would never raise anything.
        """
        for old_key, state in self._memory.items():
            if old_key == key:
                continue
            if state.label != label and not (
                self._merge_bag_labels and {state.label, label} <= {"backpack", "handbag", "suitcase"}
            ):
                continue
            last_seen = self._memory.last_seen(old_key)
            if last_seen is None or last_seen >= now:
                continue  # still being reported under its own id this frame
            if math.dist(centre, state.anchor) <= self._tolerance * state.size:
                self._memory.drop(old_key)
                return self._memory.put(key, state, now)
        return None

    def _event(self, event_type, context, track, state: _Resting, now: float, severity: Severity):
        return make_event(
            self.module_name, event_type, context, severity,
            subject={"track_id": track.track_id, "class": track.label},
            data={
                "unattended_seconds": round(now - state.unattended_since, 1),
                "threshold_seconds": self._seconds,
            },
        )

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        super().cleanup()
