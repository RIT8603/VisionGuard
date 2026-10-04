"""Zone geometry: polygons, containment and the state machine behind events.

Kept separate from the module so it can be reasoned about and tested without
a pipeline, a context or a tracker.

Polygons are stored in **normalised** coordinates (0.0-1.0 of frame width and
height) rather than pixels. ``processing.target_width`` changes the frame size,
and a camera can be reconfigured to a different resolution; a zone drawn in
pixels would silently point somewhere else after either. Normalised zones
survive both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


class ReferencePoint(str, Enum):
    """Which point of a bounding box decides whether it is inside a zone."""

    #: Where a person's feet are. Correct for zones drawn on the ground.
    BOTTOM_CENTER = "bottom_center"
    #: The box centroid. Better for overhead cameras.
    CENTER = "center"


class ZoneError(ValueError):
    """A zone definition is unusable."""


@dataclass(frozen=True)
class Zone:
    """A named polygon in normalised frame coordinates."""

    name: str
    polygon: Tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ZoneError(
                f"Zone '{self.name}' needs at least 3 points, got {len(self.polygon)}"
            )
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ZoneError(
                    f"Zone '{self.name}' point ({x}, {y}) is outside 0.0-1.0. "
                    "Polygons are normalised fractions of the frame, not pixels."
                )

    def to_pixels(self, width: int, height: int) -> np.ndarray:
        """Project onto a frame of the given size."""
        return np.array(
            [(int(round(x * width)), int(round(y * height))) for x, y in self.polygon],
            dtype=np.int32,
        )

    def contains(self, point: Point, width: int, height: int) -> bool:
        """Is a pixel point inside this zone? Points on the edge count as in."""
        polygon = self.to_pixels(width, height)
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    @classmethod
    def from_config(cls, raw: Dict) -> "Zone":
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ZoneError("Every zone needs a 'name'")

        points = raw.get("polygon")
        if not isinstance(points, (list, tuple)):
            raise ZoneError(f"Zone '{name}' needs a 'polygon' list of [x, y] points")

        parsed: List[Point] = []
        for item in points:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ZoneError(
                    f"Zone '{name}' polygon entries must be [x, y] pairs, got {item!r}"
                )
            parsed.append((float(item[0]), float(item[1])))

        return cls(name=name, polygon=tuple(parsed))


def reference_point(
    bbox, mode: ReferencePoint = ReferencePoint.BOTTOM_CENTER
) -> Point:
    """The point of ``bbox`` used for zone containment."""
    if mode is ReferencePoint.CENTER:
        return bbox.center
    cx, _ = bbox.center
    return (cx, bbox.y2)


class Occupancy(str, Enum):
    """What just happened to one track in one zone."""

    ENTERED = "entered"
    STILL_INSIDE = "still_inside"
    EXITED = "exited"
    OUTSIDE = "outside"


@dataclass
class _TrackZoneState:
    """How one track relates to one zone, over time."""

    inside: bool = False
    #: Consecutive frames the containment test has agreed with itself.
    streak: int = 0
    frames_inside: int = 0
    entered_at: float = 0.0
    dwell_reported: bool = False


#: Identifies one tracked thing. Track ids are namespaced per tracking module,
#: so ``person_tracking``'s #1 and ``vehicle_tracking``'s #1 are different
#: things and must not share state.
SubjectKey = Tuple[str, int]


def subject_key(source_module: str, track_id: int) -> SubjectKey:
    return (source_module, int(track_id))


@dataclass
class ZoneStateMachine:
    """Turns per-frame containment into entry / exit / dwell transitions.

    Two things stop this from emitting noise:

    * **Debounce.** A track must test the same way for ``min_frames`` frames
      before the state flips, so a box jittering on the boundary does not
      produce an entry/exit pair every frame.
    * **Dwell fires once.** Crossing the dwell threshold reports once per
      visit, not on every subsequent frame.
    """

    min_frames: int = 2
    states: Dict[Tuple[str, SubjectKey], _TrackZoneState] = field(default_factory=dict)

    def update(
        self, zone_name: str, subject: SubjectKey, is_inside: bool, now: float
    ) -> Tuple[Occupancy, _TrackZoneState]:
        key = (zone_name, subject)
        state = self.states.setdefault(key, _TrackZoneState())

        if is_inside == state.inside:
            # Agreement with the settled state; reset any pending flip.
            state.streak = 0
            if state.inside:
                state.frames_inside += 1
                return Occupancy.STILL_INSIDE, state
            return Occupancy.OUTSIDE, state

        state.streak += 1
        if state.streak < self.min_frames:
            # Not yet convinced; report the settled state.
            if state.inside:
                state.frames_inside += 1
                return Occupancy.STILL_INSIDE, state
            return Occupancy.OUTSIDE, state

        # The flip is confirmed.
        state.streak = 0
        state.inside = is_inside

        if is_inside:
            state.frames_inside = 1
            state.entered_at = now
            state.dwell_reported = False
            return Occupancy.ENTERED, state

        return Occupancy.EXITED, state

    def forget(self, subject: SubjectKey) -> List[Tuple[str, _TrackZoneState]]:
        """Drop a vanished track, reporting any zones it was still inside."""
        abandoned: List[Tuple[str, _TrackZoneState]] = []
        for key in [k for k in self.states if k[1] == subject]:
            state = self.states.pop(key)
            if state.inside:
                abandoned.append((key[0], state))
        return abandoned

    def transfer(self, old: SubjectKey, new: SubjectKey) -> None:
        """Carry ``old``'s state in every zone over to ``new``.

        For a tracker that lost someone and picked them up again under a new
        id: the visit continues instead of ending and starting again.
        """
        for key in [k for k in self.states if k[1] == old]:
            self.states[(key[0], new)] = self.states.pop(key)

    def is_inside(self, zone_name: str, subject: SubjectKey) -> bool:
        state = self.states.get((zone_name, subject))
        return bool(state and state.inside)

    def known_subjects(self) -> set[SubjectKey]:
        return {subject for _, subject in self.states}

    def occupancy(self, zone_name: str) -> int:
        """How many tracks are currently inside a zone."""
        return sum(
            1
            for (name, _), state in self.states.items()
            if name == zone_name and state.inside
        )

    def reset(self) -> None:
        self.states.clear()


def parse_zones(raw_zones: Sequence[Dict]) -> List[Zone]:
    """Build zones from the ``zones:`` config list."""
    if not raw_zones:
        raise ZoneError(
            "No zones configured. Add at least one to "
            "modules.restricted_zone.zones, for example:\n"
            "    zones:\n"
            "      - name: loading-bay\n"
            "        polygon: [[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]]"
        )

    zones = [Zone.from_config(raw) for raw in raw_zones]

    names = [zone.name for zone in zones]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ZoneError(f"Duplicate zone names: {sorted(duplicates)}")

    return zones
