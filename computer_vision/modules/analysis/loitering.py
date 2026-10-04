"""Loitering module.

One responsibility: report a tracked person who stays in roughly the same
place for longer than a configured time.

"The same place" is a circle around where the person was first seen standing.
Walking out of that circle restarts the clock, so someone crossing the frame
slowly is not loitering, and someone pacing a few steps back and forth is.

It runs no model and it does not decide that lingering is wrong - a guard at
a door lingers all day. It publishes the fact; what to do about it is a rule
for whoever consumes the event.

``restricted_zone`` has a dwell timer too, but that one fires only inside a
drawn zone. This one works anywhere in the frame, optionally limited to zones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.analysis.common import (
    SubjectMemory,
    gather_tracks,
    make_event,
    parse_optional_zones,
    parse_severity,
    positive_number,
    zone_containing,
)
from computer_vision.modules.analysis.geometry import ReferencePoint, reference_point, subject_key
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType

DEFAULT_SOURCES = ("person_tracking",)
EVENT_LOITERING = "loitering"


@dataclass
class _Linger:
    anchor: Tuple[float, float]
    since: float
    reported_at: float | None = None
    zone: str = ""


class LoiteringModule(BaseAIModule):
    """Emits ``loitering`` when a track stays put too long.

    Config block::

        loitering:
          enabled: true
          sources: [person_tracking]
          seconds: 30            # how long counts as loitering
          radius: 0.08           # "same place": fraction of the frame diagonal
          cooldown_seconds: 60   # repeat the event for the same person after this
          severity: warning
          zones: []              # optional; empty = anywhere in the frame
    """

    module_name = "loitering"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = DEFAULT_SOURCES

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        return tuple(str(name) for name in ((params or {}).get("sources") or DEFAULT_SOURCES))

    def initialize(self) -> None:
        self._sources = list(self.required_modules(self.params))
        self._seconds = positive_number(self.params, "seconds", 30.0)
        self._radius = positive_number(self.params, "radius", 0.08)
        self._cooldown = positive_number(self.params, "cooldown_seconds", 60.0, allow_zero=True)
        self._severity = parse_severity(self.params, Severity.WARNING)
        self._zones = parse_optional_zones(self.params.get("zones"))
        self._memory: SubjectMemory[_Linger] = SubjectMemory(forget_seconds=5.0)
        self._total = 0
        self.log.info(
            "Ready: watching %s, loitering after %.0fs within %.0f%% of the frame diagonal",
            self._sources, self._seconds, self._radius * 100,
        )

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        radius_px = self._radius * math.hypot(context.width, context.height)
        events = []
        lingering: List[int] = []

        for track in gather_tracks(context, self._sources):
            key = subject_key(track.source_module, track.track_id)
            point = reference_point(track.bbox, ReferencePoint.BOTTOM_CENTER)
            zone = zone_containing(self._zones, point, context)
            state = self._memory.get(key)

            if zone is None:
                # Outside every configured zone: forget any clock it had, so
                # time spent elsewhere never counts towards loitering here.
                self._memory.drop(key)
                continue

            if state is None or math.dist(point, state.anchor) > radius_px:
                self._memory.put(key, _Linger(anchor=point, since=now, zone=zone), now)
                continue

            self._memory.touch(key, now)
            stayed = now - state.since
            if stayed < self._seconds:
                continue

            lingering.append(track.track_id)
            due = state.reported_at is None or (
                self._cooldown > 0 and now - state.reported_at >= self._cooldown
            )
            if due:
                state.reported_at = now
                events.append(
                    make_event(
                        self.module_name, EVENT_LOITERING, context, self._severity,
                        subject={"track_id": track.track_id, "class": track.label},
                        data={
                            "seconds": round(stayed, 1),
                            "threshold_seconds": self._seconds,
                            "zone": state.zone,
                        },
                    )
                )

        self._memory.expire(now)
        for event in events:
            context.add_event(event)
            self.log.info("%s (%.0fs)", event.describe(), event.data["seconds"])
        self._total += len(events)

        return self.make_result(
            context,
            data={
                "watching": len(self._memory),
                "loitering": lingering,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        super().cleanup()
