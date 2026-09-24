"""Restricted Zone module.

One responsibility: report when a tracked thing enters, stays in, or leaves a
configured area.

This is the first module that produces **events** rather than detections. It
runs no model and detects nothing - it consumes tracks that already exist and
applies a geometric rule to them (Module_spec.md, "Restricted Zone").

It deliberately stops at publishing events. It does not notify anyone, write
an incident, or decide that an entry is a breach rather than a delivery
driver doing their job. Those are downstream decisions (Agent.md §16).

It consumes *tracks*, not detections, on purpose: without a persistent id you
cannot tell "one person entered and stayed for a minute" from "sixty separate
entries", and every event would fire once per frame.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

from app.core.context import Event, FrameContext, Overlay, Severity
from app.modules.analysis.geometry import (
    Occupancy,
    ReferencePoint,
    Zone,
    ZoneError,
    ZoneStateMachine,
    parse_zones,
    reference_point,
    subject_key,
)
from app.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_SOURCES = ("person_tracking",)
DEFAULT_MIN_FRAMES = 2

EVENT_ENTRY = "zone_entry"
EVENT_EXIT = "zone_exit"
EVENT_DWELL = "zone_dwell"

_COLOR_IDLE = (0, 210, 255)      # amber
_COLOR_OCCUPIED = (80, 80, 255)  # red


class RestrictedZoneModule(BaseAIModule):
    """Emits entry / dwell / exit events for tracks inside configured areas.

    Config block::

        restricted_zone:
          enabled: true
          sources: [person_tracking]     # any tracking modules
          reference_point: bottom_center # bottom_center | center
          min_frames: 2                  # debounce before a state flips
          dwell_seconds: 10              # 0 disables dwell events
          severity: warning
          classes: []                    # optional label allowlist
          zones:
            - name: loading-bay
              polygon: [[0.05, 0.55], [0.95, 0.55], [0.95, 0.98], [0.05, 0.98]]

    Polygons are fractions of the frame (0.0-1.0), not pixels, so they survive
    a resolution or ``target_width`` change.
    """

    module_name = "restricted_zone"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    #: Static default; the real list comes from config via required_modules().
    requires = DEFAULT_SOURCES

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        """Whichever tracking modules the configuration says to watch."""
        sources = (params or {}).get("sources") or DEFAULT_SOURCES
        return tuple(str(name) for name in sources)

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._sources = list(self.required_modules(self.params))
        self._zones: List[Zone] = []
        self._machine = ZoneStateMachine(
            min_frames=max(1, int(self.params.get("min_frames", DEFAULT_MIN_FRAMES)))
        )
        self._dwell_seconds = float(self.params.get("dwell_seconds", 0.0))
        self._classes = tuple(self.params.get("classes") or ())
        self._event_counts: Dict[str, int] = {}

        raw_severity = str(self.params.get("severity", Severity.WARNING.value)).lower()
        try:
            self._severity = Severity(raw_severity)
        except ValueError:
            valid = [s.value for s in Severity]
            raise ValueError(f"severity must be one of {valid}, got {raw_severity!r}")

        raw_reference = str(
            self.params.get("reference_point", ReferencePoint.BOTTOM_CENTER.value)
        ).lower()
        try:
            self._reference = ReferencePoint(raw_reference)
        except ValueError:
            valid = [r.value for r in ReferencePoint]
            raise ValueError(
                f"reference_point must be one of {valid}, got {raw_reference!r}"
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Parse zones. No model, no weights."""
        try:
            self._zones = parse_zones(self.params.get("zones") or [])
        except ZoneError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        self.log.info(
            "Ready: %d zone(s) %s, watching %s, reference=%s, dwell=%s",
            len(self._zones),
            [zone.name for zone in self._zones],
            self._sources,
            self._reference.value,
            f"{self._dwell_seconds}s" if self._dwell_seconds > 0 else "off",
        )

    def cleanup(self) -> None:
        self._machine.reset()
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        tracks = self._relevant_tracks(context)
        # Not `or time.perf_counter()`: a legitimate timestamp of 0.0 is
        # falsy, and falling back to wall-clock there corrupts every dwell.
        now = context.monotonic_ts

        events: List[Event] = []
        events.extend(self._retire_vanished(tracks, context, now))

        for track in tracks:
            point = reference_point(track.bbox, self._reference)
            subject = subject_key(track.source_module, track.track_id)
            for zone in self._zones:
                inside = zone.contains(point, context.width, context.height)
                transition, state = self._machine.update(zone.name, subject, inside, now)
                events.extend(
                    self._events_for(transition, state, zone, track, context, now)
                )

        for event in events:
            context.add_event(event)
            self._event_counts[event.event_type] = (
                self._event_counts.get(event.event_type, 0) + 1
            )
            self.log.info("%s", event.describe())

        occupancy = {zone.name: self._machine.occupancy(zone.name) for zone in self._zones}
        self._draw_zones(context, occupancy)

        return self.make_result(
            context,
            data={
                "zones": occupancy,
                "occupied": [name for name, count in occupancy.items() if count],
                "events": [event.to_dict() for event in events],
                "event_totals": dict(self._event_counts),
                "tracks_watched": len(tracks),
            },
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _relevant_tracks(self, context: FrameContext) -> List[Any]:
        tracks: List[Any] = []
        for source in self._sources:
            for track in context.get_tracks(source):
                if self._classes and track.label not in self._classes:
                    continue
                tracks.append(track)
        return tracks

    def _retire_vanished(
        self, tracks: Sequence[Any], context: FrameContext, now: float
    ) -> List[Event]:
        """Close out tracks that disappeared while still inside a zone.

        Without this, someone who walks in and is then lost by the tracker
        stays counted as present forever.
        """
        present = {
            subject_key(track.source_module, track.track_id) for track in tracks
        }
        events: List[Event] = []

        for subject in self._machine.known_subjects() - present:
            for zone_name, state in self._machine.forget(subject):
                events.append(
                    self._make_event(
                        EVENT_EXIT,
                        context,
                        track_id=subject[1],
                        label="",
                        zone=zone_name,
                        extra={
                            "dwell_seconds": round(max(0.0, now - state.entered_at), 2),
                            "frames_inside": state.frames_inside,
                            "reason": "track_lost",
                        },
                    )
                )
        return events

    def _events_for(
        self, transition: Occupancy, state, zone: Zone, track, context, now: float
    ) -> List[Event]:
        if transition is Occupancy.ENTERED:
            return [
                self._make_event(
                    EVENT_ENTRY, context, track.track_id, track.label, zone.name,
                    {"confidence": round(float(track.confidence), 4)},
                )
            ]

        if transition is Occupancy.EXITED:
            return [
                self._make_event(
                    EVENT_EXIT, context, track.track_id, track.label, zone.name,
                    {
                        "dwell_seconds": round(max(0.0, now - state.entered_at), 2),
                        "frames_inside": state.frames_inside,
                    },
                )
            ]

        if transition is Occupancy.STILL_INSIDE and self._dwell_seconds > 0:
            dwell = now - state.entered_at
            if dwell >= self._dwell_seconds and not state.dwell_reported:
                state.dwell_reported = True
                return [
                    self._make_event(
                        EVENT_DWELL, context, track.track_id, track.label, zone.name,
                        {
                            "dwell_seconds": round(dwell, 2),
                            "threshold_seconds": self._dwell_seconds,
                        },
                    )
                ]

        return []

    def _make_event(
        self,
        event_type: str,
        context: FrameContext,
        track_id: int,
        label: str,
        zone: str,
        extra: Dict[str, Any] | None = None,
    ) -> Event:
        subject: Dict[str, Any] = {"track_id": track_id}
        if label:
            subject["class"] = label

        return Event(
            event_type=event_type,
            source_module=self.module_name,
            frame_id=context.frame_id,
            timestamp=context.timestamp_iso,
            severity=self._severity,
            subject=subject,
            data={"zone": zone, **(extra or {})},
        )

    def _draw_zones(self, context: FrameContext, occupancy: Dict[str, int]) -> None:
        """Publish the zones for the output layer to render."""
        for zone in self._zones:
            count = occupancy.get(zone.name, 0)
            points = zone.to_pixels(context.width, context.height)
            context.add_overlay(
                Overlay(
                    kind="polygon",
                    points=[(int(x), int(y)) for x, y in points],
                    label=f"{zone.name} ({count})" if count else zone.name,
                    color=_COLOR_OCCUPIED if count else _COLOR_IDLE,
                    active=bool(count),
                )
            )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def zone_names(self) -> List[str]:
        return [zone.name for zone in self._zones]

    @property
    def event_totals(self) -> Dict[str, int]:
        return dict(self._event_counts)
