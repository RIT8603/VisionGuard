"""Attendance module.

One responsibility: turn "alice's face was recognised" into "alice arrived"
and "alice left".

It is **not** part of face recognition (Agent.md §14). Recognition answers a
question about one frame; attendance is a statement about a span of time, and
needs state recognition must not hold.

The two rules that make it usable:

* **Arrival needs corroboration.** One frame of a recognised face is not an
  arrival - recognition has a false-positive rate, and a mistaken check-in is
  worse than a late one. ``min_sightings`` frames are required.
* **Departure is an absence, not an event.** Nobody emits "left the frame".
  A check-out fires when someone has not been seen for ``absence_timeout``,
  which must be long enough to survive them turning around or being briefly
  occluded.

It publishes events and holds state for the current run. Persisting a
timesheet, computing hours worked and reporting to a payroll system are
downstream concerns that consume these events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.context import Event, FrameContext, Severity
from app.modules.base import BaseAIModule, ModuleResult, ModuleType

EVENT_CHECK_IN = "attendance_check_in"
EVENT_CHECK_OUT = "attendance_check_out"

DEFAULT_MIN_SIGHTINGS = 3
DEFAULT_ABSENCE_TIMEOUT_SEC = 60.0


@dataclass
class AttendanceRecord:
    """One identity's presence during this run."""

    name: str
    first_seen: float
    last_seen: float
    sightings: int = 0
    checked_in: bool = False
    checked_in_at: Optional[float] = None
    sessions: List[Dict[str, float]] = field(default_factory=list)

    @property
    def present_seconds(self) -> float:
        """Time inside the current session. 0 when checked out."""
        if not self.checked_in or self.checked_in_at is None:
            return 0.0
        return max(0.0, self.last_seen - self.checked_in_at)

    @property
    def total_seconds(self) -> float:
        closed = sum(s["duration_seconds"] for s in self.sessions)
        return closed + self.present_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "checked_in": self.checked_in,
            "sightings": self.sightings,
            "present_seconds": round(self.present_seconds, 1),
            "total_seconds": round(self.total_seconds, 1),
            "sessions": len(self.sessions),
        }


class AttendanceModule(BaseAIModule):
    """Tracks who is present, from recognised faces.

    Config block::

        attendance:
          enabled: true
          min_sightings: 3          # frames before a check-in is recorded
          absence_timeout: 60       # seconds unseen before a check-out
          severity: info
    """

    module_name = "attendance"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("face_recognition",)

    source_module = "face_recognition"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._min_sightings = max(1, int(self.params.get("min_sightings", DEFAULT_MIN_SIGHTINGS)))
        self._absence_timeout = float(
            self.params.get("absence_timeout", DEFAULT_ABSENCE_TIMEOUT_SEC)
        )
        self._records: Dict[str, AttendanceRecord] = {}

        raw_severity = str(self.params.get("severity", Severity.INFO.value)).lower()
        try:
            self._severity = Severity(raw_severity)
        except ValueError:
            valid = [s.value for s in Severity]
            raise ValueError(f"severity must be one of {valid}, got {raw_severity!r}")

    def initialize(self) -> None:
        """Nothing to load. Attendance is bookkeeping, not inference."""
        self.log.info(
            "Ready: consuming '%s', check-in after %d sighting(s), "
            "check-out after %.0fs unseen",
            self.source_module,
            self._min_sightings,
            self._absence_timeout,
        )

    def cleanup(self) -> None:
        self._records.clear()
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        seen_now = {
            det.extra.get("identity")
            for det in context.get_detections(self.source_module)
            if det.extra.get("recognised") and det.extra.get("identity")
        }

        events: List[Event] = []
        events.extend(self._record_sightings(seen_now, context, now))
        events.extend(self._expire_absent(seen_now, context, now))

        for event in events:
            context.add_event(event)
            self.log.info("%s", event.describe())

        present = [r.name for r in self._records.values() if r.checked_in]
        return self.make_result(
            context,
            data={
                "present": sorted(present),
                "present_count": len(present),
                "known_today": len(self._records),
                "records": [r.to_dict() for r in self._records.values()],
            },
        )

    def _record_sightings(
        self, seen: set, context: FrameContext, now: float
    ) -> List[Event]:
        events: List[Event] = []

        for name in sorted(seen):
            record = self._records.get(name)
            if record is None:
                record = AttendanceRecord(name=name, first_seen=now, last_seen=now)
                self._records[name] = record

            record.last_seen = now
            record.sightings += 1

            if not record.checked_in and record.sightings >= self._min_sightings:
                record.checked_in = True
                record.checked_in_at = now
                events.append(
                    self._event(EVENT_CHECK_IN, context, name, {"sightings": record.sightings})
                )
        return events

    def _expire_absent(
        self, seen: set, context: FrameContext, now: float
    ) -> List[Event]:
        """Check out anyone not seen for longer than the timeout."""
        events: List[Event] = []

        for record in self._records.values():
            if record.name in seen or not record.checked_in:
                continue
            if (now - record.last_seen) < self._absence_timeout:
                continue

            # Explicit `is None`, not `or`: a timestamp of 0.0 is falsy, and
            # `checked_in_at or last_seen` silently reports a zero duration.
            start = record.checked_in_at if record.checked_in_at is not None else record.last_seen
            duration = max(0.0, record.last_seen - start)
            record.sessions.append(
                {
                    "start": start,
                    "end": record.last_seen,
                    "duration_seconds": duration,
                }
            )
            record.checked_in = False
            record.checked_in_at = None
            # Reset so a re-appearance needs fresh corroboration.
            record.sightings = 0

            events.append(
                self._event(
                    EVENT_CHECK_OUT,
                    context,
                    record.name,
                    {
                        "duration_seconds": round(duration, 1),
                        "unseen_seconds": round(now - record.last_seen, 1),
                    },
                )
            )
        return events

    def _event(
        self, event_type: str, context: FrameContext, name: str, data: Dict[str, Any]
    ) -> Event:
        return Event(
            event_type=event_type,
            source_module=self.module_name,
            frame_id=context.frame_id,
            timestamp=context.timestamp_iso,
            severity=self._severity,
            subject={"identity": name, "class": "person"},
            data=data,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def present(self) -> List[str]:
        return sorted(r.name for r in self._records.values() if r.checked_in)

    def record(self, name: str) -> Optional[AttendanceRecord]:
        return self._records.get(name)
