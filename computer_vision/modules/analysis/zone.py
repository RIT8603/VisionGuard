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

from computer_vision.core.context import Event, FrameContext, Overlay, Severity
from computer_vision.modules.analysis.geometry import (
    Occupancy,
    ReferencePoint,
    Zone,
    ZoneError,
    ZoneStateMachine,
    parse_zones,
    reference_point,
    subject_key,
)
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_SOURCES = ("person_tracking",)
DEFAULT_MIN_FRAMES = 2
DEFAULT_LOST_GRACE_SEC = 2.0
DEFAULT_HANDOFF_DISTANCE = 1.0

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
          min_travel: 0.5                # 0 = off (default)
          lost_grace_seconds: 2.0        # 0 = a missed frame ends the visit
          handoff_distance: 1.0          # 0 = a new id is always a new visit
          zones:
            - name: loading-bay
              polygon: [[0.05, 0.55], [0.95, 0.55], [0.95, 0.98], [0.05, 0.98]]

    Polygons are fractions of the frame (0.0-1.0), not pixels, so they survive
    a resolution or ``target_width`` change.

    ``min_travel`` ignores a track until it has moved at least that many of
    its own box-sizes from where it was first seen. A mannequin, a poster or
    a coat rack that the detector keeps calling a person sits inside a zone
    forever without moving; a person has to walk to get there. The price:
    someone already standing still inside the zone when the camera starts (or
    when they are first detected) is not reported until they move.

    ``lost_grace_seconds`` and ``handoff_distance`` keep one visit as one
    visit when the tracker falters. A tracker reports a person only on frames
    where the detector found them, and sometimes gives the same person a new
    id. Without these, every missed frame is an exit followed by an entry, a
    visit is counted several times and nobody stays long enough to dwell.

    * A track that disappears while inside is kept for ``lost_grace_seconds``
      before its exit is reported; if it comes back in that time nothing is
      reported at all. The exit, when it comes, is timed from when the track
      was last seen. The price: a real exit by disappearance (walking behind
      something, leaving the picture) is reported that much later.
    * A track that first appears already inside a zone, within
      ``handoff_distance`` box-sizes of where a track of the same kind went
      missing inside that zone, is taken to be the same visitor under a new
      id and continues that visit.
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
        self._min_travel = max(0.0, float(self.params.get("min_travel", 0.0)))
        self._lost_grace = max(
            0.0, float(self.params.get("lost_grace_seconds", DEFAULT_LOST_GRACE_SEC))
        )
        self._handoff_distance = max(
            0.0, float(self.params.get("handoff_distance", DEFAULT_HANDOFF_DISTANCE))
        )
        #: subject -> (when last seen, reference point then, label)
        self._last_seen: Dict[Any, Tuple[float, Tuple[float, float], str]] = {}
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
            "Ready: %d zone(s) %s, watching %s, reference=%s, dwell=%s, "
            "min_travel=%s, lost_grace=%s, handoff=%s",
            len(self._zones),
            [zone.name for zone in self._zones],
            self._sources,
            self._reference.value,
            f"{self._dwell_seconds}s" if self._dwell_seconds > 0 else "off",
            self._min_travel or "off",
            f"{self._lost_grace}s" if self._lost_grace > 0 else "off",
            self._handoff_distance or "off",
        )

    def cleanup(self) -> None:
        self._machine.reset()
        self._last_seen.clear()
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
        self._hand_over(tracks, context)
        events.extend(self._retire_vanished(tracks, context, now))

        for track in tracks:
            point = reference_point(track.bbox, self._reference)
            subject = subject_key(track.source_module, track.track_id)
            self._last_seen[subject] = (now, point, track.label)
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
                if self._min_travel and not self._has_moved(track):
                    continue
                tracks.append(track)
        return tracks

    def _has_moved(self, track: Any) -> bool:
        """False only for a track known to have stayed where it first appeared.

        A track that does not report its travel (another tracker's output) is
        taken at face value.
        """
        travel = getattr(track, "extra", {}).get("travel")
        return travel is None or float(travel) >= self._min_travel

    def _hand_over(self, tracks: Sequence[Any], context: FrameContext) -> None:
        """Continue a visit under a new id when the tracker renumbered someone.

        A track never seen before that appears already inside a zone, close to
        where a track of the same kind went missing inside that zone, takes
        over the missing track's visit: no exit, no second entry, and the
        dwell clock keeps running.
        """
        if not self._handoff_distance:
            return
        known = self._machine.known_subjects()
        present = {subject_key(track.source_module, track.track_id) for track in tracks}
        missing = known - present
        if not missing:
            return

        for track in tracks:
            subject = subject_key(track.source_module, track.track_id)
            if subject in known:
                continue
            point = reference_point(track.bbox, self._reference)
            reach = self._handoff_distance * max(track.bbox.width, track.bbox.height)

            best, best_distance = None, reach
            for old in missing:
                if old[0] != subject[0] or old not in self._last_seen:
                    continue
                _, old_point, old_label = self._last_seen[old]
                if old_label != track.label:
                    continue
                distance = ((point[0] - old_point[0]) ** 2 + (point[1] - old_point[1]) ** 2) ** 0.5
                if distance > best_distance:
                    continue
                if not any(
                    self._machine.is_inside(zone.name, old)
                    and zone.contains(point, context.width, context.height)
                    for zone in self._zones
                ):
                    continue
                best, best_distance = old, distance

            if best is not None:
                self._machine.transfer(best, subject)
                self._last_seen.pop(best, None)
                missing.discard(best)
                known.add(subject)

    def _retire_vanished(
        self, tracks: Sequence[Any], context: FrameContext, now: float
    ) -> List[Event]:
        """Close out tracks that disappeared while still inside a zone.

        Without this, someone who walks in and is then lost by the tracker
        stays counted as present forever. A track is given
        ``lost_grace_seconds`` to come back first: a tracker leaves a person
        out on any frame the detector missed them.
        """
        present = {
            subject_key(track.source_module, track.track_id) for track in tracks
        }
        events: List[Event] = []

        for subject in self._machine.known_subjects() - present:
            last_seen = self._last_seen.get(subject, (now, None, ""))[0]
            if self._lost_grace and 0.0 <= now - last_seen < self._lost_grace:
                continue
            self._last_seen.pop(subject, None)
            for zone_name, state in self._machine.forget(subject):
                # Timed to when the track was last seen, not to when the wait
                # for it ran out. Without a grace period that is this frame.
                left_at = last_seen if self._lost_grace else now
                events.append(
                    self._make_event(
                        EVENT_EXIT,
                        context,
                        track_id=subject[1],
                        label="",
                        zone=zone_name,
                        extra={
                            "dwell_seconds": round(max(0.0, left_at - state.entered_at), 2),
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
