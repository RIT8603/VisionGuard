"""Unknown Person module.

One responsibility: raise an event when a face is present that does not match
any enrolled identity.

It is **not a visual detector** - there is no such thing as an "unknown person"
in pixels (Agent.md §13). It is a higher-level conclusion drawn from
``face_detection`` plus ``face_recognition`` plus the identity store, and it is
exactly as reliable as those three are.

It publishes events and stops. Deciding that an unknown face warrants a phone
call is downstream logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.context import Event, FrameContext, Severity
from app.modules.base import BaseAIModule, ModuleResult, ModuleType

EVENT_UNKNOWN_PERSON = "unknown_person"
DEFAULT_COOLDOWN_SEC = 30.0
DEFAULT_MIN_SIGHTINGS = 3


class UnknownPersonModule(BaseAIModule):
    """Emits an event for faces that matched no enrolled identity.

    Config block::

        unknown_person:
          enabled: true
          min_sightings: 3       # frames before an unknown face is reported
          cooldown_seconds: 30   # quiet period after reporting
          severity: warning

    Two guards keep this from being useless in practice:

    * ``min_sightings`` - a single frame of a half-turned head is not evidence
      of an intruder, and recognition is least reliable at exactly the moments
      it briefly fails.
    * ``cooldown_seconds`` - one person standing in view must not generate an
      event per frame. Without it a live camera produces thousands an hour.
    """

    module_name = "unknown_person"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("face_recognition",)

    source_module = "face_recognition"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._min_sightings = max(1, int(self.params.get("min_sightings", DEFAULT_MIN_SIGHTINGS)))
        self._cooldown = float(self.params.get("cooldown_seconds", DEFAULT_COOLDOWN_SEC))
        self._consecutive = 0
        self._last_reported_at: float | None = None
        self._total_events = 0

        raw_severity = str(self.params.get("severity", Severity.WARNING.value)).lower()
        try:
            self._severity = Severity(raw_severity)
        except ValueError:
            valid = [s.value for s in Severity]
            raise ValueError(f"severity must be one of {valid}, got {raw_severity!r}")

    def initialize(self) -> None:
        """Nothing to load. This module is a rule, not a model."""
        self.log.info(
            "Ready: consuming '%s', report after %d sighting(s), cooldown %.0fs",
            self.source_module,
            self._min_sightings,
            self._cooldown,
        )

    def process(self, context: FrameContext) -> ModuleResult:
        unknown = [
            det
            for det in context.get_detections(self.source_module)
            if not det.extra.get("recognised", False)
        ]

        if not unknown:
            self._consecutive = 0
            return self.make_result(
                context, data={"unknown_faces": 0, "reported": False}
            )

        self._consecutive += 1
        now = context.monotonic_ts
        reported = False

        if self._consecutive >= self._min_sightings and self._ready_to_report(now):
            self._last_reported_at = now
            self._total_events += 1
            reported = True

            best = max(unknown, key=lambda d: d.extra.get("similarity", 0.0))
            event = Event(
                event_type=EVENT_UNKNOWN_PERSON,
                source_module=self.module_name,
                frame_id=context.frame_id,
                timestamp=context.timestamp_iso,
                severity=self._severity,
                subject={"class": "face", "count": len(unknown)},
                data={
                    # How close the nearest enrolment was: distinguishes
                    # "nobody like them" from "almost matched someone".
                    "closest_similarity": best.extra.get("similarity", 0.0),
                    "consecutive_sightings": self._consecutive,
                },
            )
            context.add_event(event)
            self.log.info("%s (%d face(s))", event.describe(), len(unknown))

        return self.make_result(
            context,
            data={
                "unknown_faces": len(unknown),
                "consecutive_sightings": self._consecutive,
                "reported": reported,
                "total_events": self._total_events,
            },
        )

    def _ready_to_report(self, now: float) -> bool:
        if self._last_reported_at is None:
            return True
        return (now - self._last_reported_at) >= self._cooldown

    @property
    def total_events(self) -> int:
        return self._total_events
