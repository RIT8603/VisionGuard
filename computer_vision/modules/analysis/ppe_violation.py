"""PPE Violation module.

One responsibility: report a tracked person who is missing required protective
equipment (or wears a "no-helmet"-style class) for a sustained stretch.

``ppe_detection`` only says "a helmet is here". This module pairs each PPE box
with the person it is on - by where it sits on the body, so a helmet must be
near the head and boots near the feet - and judges that against the site rule
in ``required``. It runs no model of its own.

Decisions are made per person track and over several frames: one frame where
the detector misses a helmet is not a violation, most of the recent frames
missing it is. Small, distant people are not judged at all (``min_person_height``)
- at that size a missing helmet is far more likely a detector miss.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Sequence, Tuple

from computer_vision.core.context import BBox, Detection, FrameContext, Severity
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
DEFAULT_PPE_SOURCE = "ppe_detection"
EVENT_PPE_VIOLATION = "ppe_violation"

#: Where on a person box (top 0.0 .. bottom 1.0 of its height) an item's centre
#: must lie to count as worn by that person.
BODY_REGIONS: Dict[str, Tuple[float, float]] = {
    "head": (-0.05, 0.35),
    "torso": (0.15, 0.75),
    "feet": (0.70, 1.05),
    "any": (-0.05, 1.05),
}
#: Common PPE class names and the body region they are worn on. Anything not
#: listed defaults to "any"; override per label with ``regions``.
DEFAULT_REGIONS: Dict[str, str] = {
    "helmet": "head", "hardhat": "head", "hard-hat": "head", "head_helmet": "head",
    "vest": "torso", "safety-vest": "torso", "safety_vest": "torso", "jacket": "torso",
    "boots": "feet", "shoes": "feet", "safety-boots": "feet",
    "gloves": "any", "mask": "head", "goggles": "head", "glasses": "head",
}
#: Horizontal slack: an item may stick out this fraction of the box width.
_X_MARGIN = 0.15


@dataclass
class _Person:
    #: Per judged frame: which required items were missing, and whether a
    #: violation class was seen on the person.
    missing: Deque[frozenset] = field(default_factory=deque)
    flagged: Deque[bool] = field(default_factory=deque)
    reported_at: float | None = None


class PPEViolationModule(BaseAIModule):
    """Emits ``ppe_violation`` when a person lacks required PPE for most recent frames.

    Config block::

        ppe_violation:
          enabled: true
          sources: [person_tracking]   # and ppe_detection (ppe_source)
          required: [helmet, vest]     # ppe_detection labels every person must wear
          violation_labels: []         # classes that are violations by themselves,
                                       # e.g. [no-helmet, no-vest] if the model has them
          regions: {}                  # override where a label is worn: head|torso|feet|any
          min_person_height: 80        # px in the processing frame; smaller people are skipped
          min_frames: 10               # a violation in 10...
          window_frames: 15            # ...of the person's last 15 judged frames
          cooldown_seconds: 120        # repeat for the same person after this
          severity: warning
          zones: []                    # optional: judge only inside these zones
    """

    module_name = "ppe_violation"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = (*DEFAULT_SOURCES, DEFAULT_PPE_SOURCE)

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        params = params or {}
        sources = tuple(str(name) for name in (params.get("sources") or DEFAULT_SOURCES))
        return (*sources, str(params.get("ppe_source") or DEFAULT_PPE_SOURCE))

    def initialize(self) -> None:
        required_modules = self.required_modules(self.params)
        self._sources = list(required_modules[:-1])
        self._ppe_source = required_modules[-1]
        self._required = tuple(str(label) for label in (self.params.get("required") or ()))
        self._violation_labels = frozenset(
            str(label) for label in (self.params.get("violation_labels") or ())
        )
        if not self._required and not self._violation_labels:
            raise ModuleInitializationError(
                "ppe_violation needs `required` (labels every person must wear) "
                "and/or `violation_labels` (labels that are violations by themselves)"
            )
        regions = dict(DEFAULT_REGIONS)
        for label, region in (self.params.get("regions") or {}).items():
            if region not in BODY_REGIONS:
                raise ModuleInitializationError(
                    f"ppe_violation.regions.{label}: '{region}' is not one of {sorted(BODY_REGIONS)}"
                )
            regions[str(label)] = str(region)
        self._regions = regions
        self._min_height = positive_number(self.params, "min_person_height", 80.0, allow_zero=True)
        self._min_frames = max(1, int(self.params.get("min_frames", 10)))
        self._window = max(self._min_frames, int(self.params.get("window_frames", 15)))
        self._cooldown = positive_number(self.params, "cooldown_seconds", 120.0, allow_zero=True)
        self._severity = parse_severity(self.params, Severity.WARNING)
        self._zones = parse_optional_zones(self.params.get("zones"))
        self._memory: SubjectMemory[_Person] = SubjectMemory(forget_seconds=5.0)
        self._total = 0
        self.log.info(
            "Ready: required %s, violation classes %s, judged after %d of %d frames, people >= %.0f px",
            list(self._required), sorted(self._violation_labels), self._min_frames, self._window,
            self._min_height,
        )

    # ------------------------------------------------------------------ #

    def _worn_by(self, item: BBox, label: str, person: BBox) -> bool:
        """Whether an item box sits where ``label`` is worn on this person."""
        cx, cy = item.center
        margin = _X_MARGIN * person.width
        if not (person.x1 - margin <= cx <= person.x2 + margin):
            return False
        top, bottom = BODY_REGIONS[self._regions.get(label, "any")]
        relative = (cy - person.y1) / max(person.height, 1e-6)
        return top <= relative <= bottom

    def _judge(self, person: BBox, items: Sequence[Detection]) -> Tuple[frozenset, bool]:
        worn = {item.label for item in items if self._worn_by(item.bbox, item.label, person)}
        missing = frozenset(label for label in self._required if label not in worn)
        return missing, bool(worn & self._violation_labels)

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        items = context.get_detections(self._ppe_source)
        events = []
        violating: List[int] = []

        for track in gather_tracks(context, self._sources, classes=("person",)):
            if track.time_since_update > 0 or track.bbox.height < self._min_height:
                continue  # not observed this frame, or too small to judge fairly
            zone = zone_containing(
                self._zones, reference_point(track.bbox, ReferencePoint.BOTTOM_CENTER), context
            )
            if zone is None:
                continue
            key = subject_key(track.source_module, track.track_id)
            state = self._memory.get(key)
            if state is None:
                state = self._memory.put(
                    key, _Person(deque(maxlen=self._window), deque(maxlen=self._window)), now
                )
            else:
                self._memory.touch(key, now)

            missing, flagged = self._judge(track.bbox, items)
            state.missing.append(missing)
            state.flagged.append(flagged)

            lacking = sorted(
                label for label in self._required
                if sum(label in frame for frame in state.missing) >= self._min_frames
            )
            flagged_frames = sum(state.flagged)
            if not lacking and flagged_frames < self._min_frames:
                continue

            violating.append(track.track_id)
            due = state.reported_at is None or (
                self._cooldown > 0 and now - state.reported_at >= self._cooldown
            )
            if not due:
                continue
            state.reported_at = now
            seen_violation = sorted(
                {item.label for item in items
                 if item.label in self._violation_labels and self._worn_by(item.bbox, item.label, track.bbox)}
            )
            events.append(
                make_event(
                    self.module_name, EVENT_PPE_VIOLATION, context, self._severity,
                    subject={"track_id": track.track_id, "class": track.label},
                    data={
                        "missing": lacking,
                        "violation_classes": seen_violation,
                        "frames_judged": len(state.missing),
                        "window_frames": self._window,
                        "zone": zone,
                        "identity": track.extra.get("identity") or "",
                    },
                )
            )

        self._memory.expire(now)
        for event in events:
            context.add_event(event)
            self.log.warning(
                "%s: missing %s%s", event.describe(), event.data.get("missing") or "-",
                f", wearing {event.data['violation_classes']}" if event.data.get("violation_classes") else "",
            )
        self._total += len(events)

        return self.make_result(
            context,
            data={
                "judged": len(self._memory),
                "violating": violating,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        super().cleanup()
