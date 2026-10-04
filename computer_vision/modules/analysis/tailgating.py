"""Tailgating (piggybacking) module.

One responsibility: report a person who passes a controlled entry line right
behind another person.

An access-controlled door admits one person per authorisation. When a second
person crosses the same line, in the same direction, within a couple of
seconds of the first, they most likely walked in on the first person's
access. That is what this reports.

It needs one thing from the site: a **line** across the doorway, drawn in
normalised frame coordinates. A person "crosses" when the foot point of their
track moves from one side of that line to the other, through the segment.

As a by-product it counts crossings in each direction, which is a gate
in/out counter for free.

It knows nothing about badges or door controllers, so it cannot tell an
authorised pair from an unauthorised one - two colleagues who both badged in
back to back look identical. Treat the event as "worth a look".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from computer_vision.core.context import FrameContext, Overlay, Severity
from computer_vision.modules.analysis.common import (
    SubjectMemory,
    gather_tracks,
    make_event,
    parse_severity,
    positive_number,
)
from computer_vision.modules.analysis.geometry import ReferencePoint, reference_point, subject_key
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_SOURCES = ("person_tracking",)
EVENT_TAILGATING = "tailgating"
EVENT_CROSSING = "line_crossed"
_DIRECTIONS = ("any", "in", "out")
_COLOR_LINE = (255, 200, 0)

Point = Tuple[float, float]


def _side(point: Point, a: Point, b: Point) -> float:
    """Signed distance (pixels) of ``point`` from the infinite line a->b."""
    length = math.dist(a, b) or 1.0
    return ((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])) / length


def _segments_cross(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    """Does the move p1->p2 pass through the segment a-b (not just its line)?"""
    d1, d2 = _side(p1, a, b), _side(p2, a, b)
    d3, d4 = _side(a, p1, p2), _side(b, p1, p2)
    return (d1 * d2 < 0) and (d3 * d4 <= 0)


@dataclass
class _Walker:
    #: Last point at which the track was clearly on one side of the line.
    point: Point
    side: float


class TailgatingModule(BaseAIModule):
    """Emits ``tailgating`` when two people cross the entry line back to back.

    Config block::

        tailgating:
          enabled: true
          sources: [person_tracking]
          line: [[0.30, 0.60], [0.70, 0.60]]   # across the doorway, 0.0-1.0
          min_gap_seconds: 2.0     # a second person sooner than this is tailgating
          direction: any           # any | in | out - which way counts as entering
          report_crossings: false  # also emit a line_crossed event per person
          severity: warning

    "in" is a crossing from the left side of the line to its right side when
    looking from the first point towards the second; swap the two points to
    flip it.
    """

    module_name = "tailgating"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = DEFAULT_SOURCES

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        return tuple(str(name) for name in ((params or {}).get("sources") or DEFAULT_SOURCES))

    def initialize(self) -> None:
        self._sources = list(self.required_modules(self.params))
        self._line = self._parse_line(self.params.get("line"))
        self._min_gap = positive_number(self.params, "min_gap_seconds", 2.0)
        self._severity = parse_severity(self.params, Severity.WARNING)
        self._report_crossings = bool(self.params.get("report_crossings", False))

        self._direction = str(self.params.get("direction", "any")).lower()
        if self._direction not in _DIRECTIONS:
            raise ModuleInitializationError(
                f"tailgating.direction must be one of {list(_DIRECTIONS)}, got {self._direction!r}"
            )

        self._memory: SubjectMemory[_Walker] = SubjectMemory(forget_seconds=3.0)
        #: (time, track id) of the latest crossing, per direction.
        self._last: Dict[str, Tuple[float, int]] = {}
        self._counts = {"in": 0, "out": 0}
        self._total = 0
        self.log.info(
            "Ready: line %s, tailgating if a second person follows within %.1fs (%s)",
            self._line, self._min_gap, self._direction,
        )

    @staticmethod
    def _parse_line(raw: Any) -> Tuple[Point, Point]:
        ok = (
            isinstance(raw, (list, tuple))
            and len(raw) == 2
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw)
        )
        if not ok:
            raise ModuleInitializationError(
                "tailgating needs a 'line' across the entry, as two [x, y] points in "
                "0.0-1.0 frame fractions, for example:\n"
                "    line: [[0.30, 0.60], [0.70, 0.60]]"
            )
        a, b = (float(raw[0][0]), float(raw[0][1])), (float(raw[1][0]), float(raw[1][1]))
        if not all(0.0 <= v <= 1.0 for v in (*a, *b)):
            raise ModuleInitializationError(
                "tailgating.line points must be 0.0-1.0 fractions of the frame, not pixels"
            )
        if a == b:
            raise ModuleInitializationError("tailgating.line needs two different points")
        return a, b

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        a = (self._line[0][0] * context.width, self._line[0][1] * context.height)
        b = (self._line[1][0] * context.width, self._line[1][1] * context.height)
        # A foot point hovering on the line must not count as crossing back
        # and forth; inside this band the last definite side is kept.
        dead_band = 0.01 * math.hypot(context.width, context.height)
        events = []

        for track in gather_tracks(context, self._sources):
            key = subject_key(track.source_module, track.track_id)
            point = reference_point(track.bbox, ReferencePoint.BOTTOM_CENTER)
            side = _side(point, a, b)
            state = self._memory.get(key)

            if abs(side) < dead_band:
                if state is not None:
                    self._memory.touch(key, now)
                continue
            if state is None:
                self._memory.put(key, _Walker(point=point, side=side), now)
                continue

            crossed = (side * state.side < 0) and _segments_cross(state.point, point, a, b)
            previous_side = state.side
            self._memory.put(key, _Walker(point=point, side=side), now)
            if not crossed:
                continue

            direction = "in" if previous_side < 0 else "out"
            self._counts[direction] += 1
            if self._report_crossings:
                events.append(
                    make_event(
                        self.module_name, EVENT_CROSSING, context, Severity.INFO,
                        subject={"track_id": track.track_id, "class": track.label},
                        data={"direction": direction, "count": self._counts[direction]},
                    )
                )

            earlier = self._last.get(direction)
            self._last[direction] = (now, track.track_id)
            if self._direction not in ("any", direction) or earlier is None:
                continue
            gap = now - earlier[0]
            if earlier[1] != track.track_id and gap <= self._min_gap:
                events.append(
                    make_event(
                        self.module_name, EVENT_TAILGATING, context, self._severity,
                        subject={"track_id": track.track_id, "class": track.label},
                        data={
                            "follows_track_id": earlier[1],
                            "gap_seconds": round(gap, 2),
                            "threshold_seconds": self._min_gap,
                            "direction": direction,
                        },
                    )
                )

        self._memory.expire(now)
        for event in events:
            context.add_event(event)
            if event.event_type == EVENT_TAILGATING:
                self._total += 1
                self.log.info(
                    "%s behind #%s (%.1fs)", event.describe(),
                    event.data["follows_track_id"], event.data["gap_seconds"],
                )

        context.add_overlay(
            Overlay(
                kind="line",
                points=[(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))],
                label=f"in {self._counts['in']} / out {self._counts['out']}",
                color=_COLOR_LINE,
            )
        )
        return self.make_result(
            context,
            data={
                "crossings": dict(self._counts),
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    @property
    def crossings(self) -> Dict[str, int]:
        return dict(self._counts)

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        super().cleanup()
