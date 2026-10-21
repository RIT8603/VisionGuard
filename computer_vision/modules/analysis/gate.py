"""Entry/exit gate counter.

One responsibility: when a tracked person crosses the gate line, record that
they **entered** or **exited**, and who they were if their face was recognised.

* **Line + direction.** The gate is a line in 0.0-1.0 frame fractions. A person
  crosses when the foot point of their track moves from one side of the line to
  the other *through the segment* (shared geometry with ``tailgating``). Which
  side is "inside" is configurable, so the same line works for any camera angle.
* **Stable IDs.** Crossings are judged per person *track*, so a person counts
  once however many frames they spend near the line, and a foot point hovering
  on the line does not count back and forth (a dead band keeps the last side).
* **Who.** ``face_recognition`` attaches recognised names to person tracks
  (``person_track_id``). The gate remembers the last name seen on each track,
  so a person recognised while walking up is named when they cross later, even
  with their back to the camera by then.

Emits ``gate_entry`` / ``gate_exit`` events (logged durably by the dashboard)
and keeps live counts: entered, exited, and currently inside.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from computer_vision.core.context import FrameContext, Overlay, Severity
from computer_vision.modules.analysis.common import SubjectMemory, gather_tracks, make_event, parse_severity
from computer_vision.modules.analysis.geometry import ReferencePoint, reference_point, subject_key
from computer_vision.modules.analysis.tailgating import _segments_cross, _side
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_SOURCES = ("person_tracking",)
EVENT_ENTRY = "gate_entry"
EVENT_EXIT = "gate_exit"
_COLOR = (80, 220, 120)

Point = Tuple[float, float]


class GateCounterModule(BaseAIModule):
    """Counts people entering and leaving through a gate line.

    Config block::

        gate_counter:
          enabled: true
          line: [[0.10, 0.65], [0.90, 0.65]]   # across the gate, 0.0-1.0
          inside: right        # which side of the line (looking from the first
                               # point to the second) is inside: left | right
          name: Main gate
          severity: info

    Swap ``inside`` if entries and exits come out the wrong way round.
    """

    module_name = "gate_counter"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = DEFAULT_SOURCES
    #: Use this frame's recognised names when face recognition is on.
    run_after = ("face_recognition",)

    def initialize(self) -> None:
        self._line = self._parse_line(self.params.get("line"))
        inside = str(self.params.get("inside", "right")).lower()
        if inside not in ("left", "right"):
            raise ModuleInitializationError("gate_counter.inside must be 'left' or 'right'")
        # _side() is positive on the right of a->b in image coordinates.
        self._inside_sign = 1.0 if inside == "right" else -1.0
        self._name = str(self.params.get("name", "Gate"))
        self._severity = parse_severity(self.params, Severity.INFO)
        self._memory: SubjectMemory[Tuple[Point, float]] = SubjectMemory(forget_seconds=3.0)
        #: person track id -> last recognised name (kept a while after the face is gone).
        self._names: Dict[int, Tuple[str, float]] = {}
        self._entered = 0
        self._exited = 0
        self.log.info("Ready: gate '%s' line %s, inside on the %s", self._name, self._line, inside)

    @staticmethod
    def _parse_line(raw: Any) -> Tuple[Point, Point]:
        ok = (
            isinstance(raw, (list, tuple))
            and len(raw) == 2
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw)
        )
        if not ok:
            raise ModuleInitializationError(
                "gate_counter needs a 'line' across the gate as two [x, y] points in "
                "0.0-1.0 frame fractions, e.g. line: [[0.10, 0.65], [0.90, 0.65]]"
            )
        a, b = (float(raw[0][0]), float(raw[0][1])), (float(raw[1][0]), float(raw[1][1]))
        if not all(0.0 <= v <= 1.0 for v in (*a, *b)) or a == b:
            raise ModuleInitializationError("gate_counter.line must be two different 0.0-1.0 points")
        return a, b

    # ------------------------------------------------------------------ #

    def _learn_names(self, context: FrameContext, now: float) -> None:
        for det in context.get_detections("face_recognition"):
            track_id = det.extra.get("person_track_id")
            identity = det.extra.get("identity")
            if track_id is not None and identity:
                self._names[int(track_id)] = (str(identity), now)
        # Forget names of people long gone, so the map cannot grow forever.
        for track_id in [t for t, (_, seen) in self._names.items() if now - seen > 120.0]:
            del self._names[track_id]

    def _name_of(self, track_id: int) -> Optional[str]:
        entry = self._names.get(track_id)
        return entry[0] if entry else None

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        self._learn_names(context, now)
        a = (self._line[0][0] * context.width, self._line[0][1] * context.height)
        b = (self._line[1][0] * context.width, self._line[1][1] * context.height)
        dead_band = 0.01 * math.hypot(context.width, context.height)
        events = []

        for track in gather_tracks(context, DEFAULT_SOURCES):
            key = subject_key(track.source_module, track.track_id)
            point = reference_point(track.bbox, ReferencePoint.BOTTOM_CENTER)
            side = _side(point, a, b)
            state = self._memory.get(key)

            if abs(side) < dead_band:
                if state is not None:
                    self._memory.touch(key, now)
                continue
            self._memory.put(key, (point, side), now)
            if state is None:
                continue
            last_point, last_side = state
            if side * last_side >= 0 or not _segments_cross(last_point, point, a, b):
                continue

            entering = side * self._inside_sign > 0
            if entering:
                self._entered += 1
            else:
                self._exited += 1
            identity = self._name_of(track.track_id)
            events.append(
                make_event(
                    self.module_name,
                    EVENT_ENTRY if entering else EVENT_EXIT,
                    context,
                    self._severity,
                    subject={"track_id": track.track_id, "class": track.label},
                    data={
                        "gate": self._name,
                        "direction": "in" if entering else "out",
                        "identity": identity or "unknown",
                        "inside": self.inside,
                    },
                )
            )

        self._memory.expire(now)
        for event in events:
            context.add_event(event)
            self.log.info("%s: %s #%s (%s)", event.event_type, event.data["identity"],
                          event.subject["track_id"], self._name)

        context.add_overlay(
            Overlay(
                kind="line",
                points=[(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))],
                label=f"{self._name}: in {self._entered} / out {self._exited} / inside {self.inside}",
                color=_COLOR,
            )
        )
        return self.make_result(
            context,
            data={
                "gate": self._name,
                "entered": self._entered,
                "exited": self._exited,
                "inside": self.inside,
                "events": [event.to_dict() for event in events],
            },
        )

    @property
    def inside(self) -> int:
        """People who entered and have not left. Never below zero: someone who
        was already inside when counting started can still walk out."""
        return max(0, self._entered - self._exited)

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        super().cleanup()
