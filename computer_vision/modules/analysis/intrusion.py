"""Intrusion Detection module.

One responsibility: report a person present where, and **when**, nobody
should be.

``restricted_zone`` answers "is someone in this area?". This adds the clock:
a corridor that is busy all day is an incident at 02:00. An intrusion is a
tracked person inside the watched area during the configured hours.

This is the part of "illegal activity detection" that can be decided from
geometry and time alone. It does not recognise *what* a person is doing -
fighting, theft and vandalism need an action-recognition model this project
does not have, and nothing here pretends otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as clock_time
from typing import Any, Dict, List, Sequence, Tuple

from computer_vision.core.context import FrameContext, Overlay, Severity
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
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_SOURCES = ("person_tracking",)
EVENT_INTRUSION = "intrusion"
_COLOR_ARMED = (60, 60, 255)
_COLOR_IDLE = (150, 150, 150)

Window = Tuple[clock_time, clock_time]


def parse_hours(raw: Any) -> List[Window]:
    """``"22:00-06:00"`` or a list of such ranges -> ``[(start, end), ...]``.

    An empty value means "always". A range whose end is before its start
    wraps past midnight.
    """
    if raw in (None, "", []):
        return []
    items: Sequence[Any] = [raw] if isinstance(raw, str) else raw
    windows: List[Window] = []
    for item in items:
        try:
            start_text, end_text = str(item).split("-")
            start = datetime.strptime(start_text.strip(), "%H:%M").time()
            end = datetime.strptime(end_text.strip(), "%H:%M").time()
        except ValueError:
            raise ModuleInitializationError(
                f"active_hours entries must look like '22:00-06:00', got {item!r}"
            ) from None
        windows.append((start, end))
    return windows


def within_hours(windows: Sequence[Window], moment: clock_time) -> bool:
    if not windows:
        return True
    for start, end in windows:
        if start <= end:
            if start <= moment < end:
                return True
        elif moment >= start or moment < end:  # wraps midnight
            return True
    return False


@dataclass
class _Presence:
    frames: int = 0
    reported_at: float | None = None


class IntrusionDetectionModule(BaseAIModule):
    """Emits ``intrusion`` for a person in the watched area during set hours.

    Config block::

        intrusion_detection:
          enabled: true
          sources: [person_tracking]
          active_hours: "22:00-06:00"   # or a list of ranges; empty = always
          min_frames: 3                 # frames of presence before reporting
          cooldown_seconds: 60          # repeat for the same person after this
          severity: critical
          zones: []                     # optional; empty = the whole frame

    Hours are the local time of the machine running VisionGuard.
    """

    module_name = "intrusion_detection"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = DEFAULT_SOURCES

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        return tuple(str(name) for name in ((params or {}).get("sources") or DEFAULT_SOURCES))

    def initialize(self) -> None:
        self._sources = list(self.required_modules(self.params))
        self._hours = parse_hours(self.params.get("active_hours"))
        self._min_frames = max(1, int(self.params.get("min_frames", 3)))
        self._cooldown = positive_number(self.params, "cooldown_seconds", 60.0, allow_zero=True)
        self._severity = parse_severity(self.params, Severity.CRITICAL)
        self._zones = parse_optional_zones(self.params.get("zones"))
        self._memory: SubjectMemory[_Presence] = SubjectMemory(forget_seconds=5.0)
        self._total = 0
        self.log.info(
            "Ready: watching %s, %s, hours=%s",
            self._sources,
            f"{len(self._zones)} zone(s)" if self._zones else "whole frame",
            self.params.get("active_hours") or "always",
        )

    def is_armed(self, context: FrameContext) -> bool:
        """Is this frame inside the configured hours (local time)?"""
        return within_hours(self._hours, context.wall_time.astimezone().time())

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        armed = self.is_armed(context)
        events = []
        intruders: List[int] = []

        if armed:
            for track in gather_tracks(context, self._sources):
                point = reference_point(track.bbox, ReferencePoint.BOTTOM_CENTER)
                zone = zone_containing(self._zones, point, context)
                key = subject_key(track.source_module, track.track_id)
                if zone is None:
                    self._memory.drop(key)
                    continue

                state = self._memory.get(key) or _Presence()
                state.frames += 1
                self._memory.put(key, state, now)
                if state.frames < self._min_frames:
                    continue

                intruders.append(track.track_id)
                due = state.reported_at is None or (
                    self._cooldown > 0 and now - state.reported_at >= self._cooldown
                )
                if due:
                    state.reported_at = now
                    events.append(
                        make_event(
                            self.module_name, EVENT_INTRUSION, context, self._severity,
                            subject={"track_id": track.track_id, "class": track.label},
                            data={
                                "zone": zone,
                                "local_time": context.wall_time.astimezone().strftime("%H:%M:%S"),
                            },
                        )
                    )
            self._memory.expire(now)
        else:
            self._memory.clear()

        for event in events:
            context.add_event(event)
            self.log.info("%s", event.describe())
        self._total += len(events)

        for zone in self._zones:
            context.add_overlay(
                Overlay(
                    kind="polygon",
                    points=[(int(x), int(y)) for x, y in zone.to_pixels(context.width, context.height)],
                    label=f"{zone.name} (armed)" if armed else zone.name,
                    color=_COLOR_ARMED if armed else _COLOR_IDLE,
                    active=bool(intruders),
                )
            )

        return self.make_result(
            context,
            data={
                "armed": armed,
                "intruders": intruders,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        super().cleanup()
