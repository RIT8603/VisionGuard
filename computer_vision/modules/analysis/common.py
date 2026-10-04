"""Small pieces every event-producing analysis module needs.

Kept deliberately thin: configuration parsing that fails with a readable
message, one way to build an :class:`Event`, and per-subject state that is
forgotten once the tracker stops reporting the subject.
"""

from __future__ import annotations

from typing import Any, Dict, Generic, Iterable, List, Sequence, Tuple, TypeVar

from computer_vision.core.context import Event, FrameContext, Severity
from computer_vision.modules.analysis.geometry import Zone, ZoneError
from computer_vision.modules.base import ModuleInitializationError

State = TypeVar("State")
SubjectKey = Tuple[str, int]


def parse_severity(params: Dict[str, Any], default: Severity) -> Severity:
    raw = str(params.get("severity", default.value)).lower()
    try:
        return Severity(raw)
    except ValueError:
        valid = [level.value for level in Severity]
        raise ModuleInitializationError(
            f"severity must be one of {valid}, got {raw!r}"
        ) from None


def positive_number(params: Dict[str, Any], key: str, default: float, *, allow_zero: bool = False) -> float:
    """A float parameter that must be > 0 (or >= 0 with ``allow_zero``)."""
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        raise ModuleInitializationError(f"{key} must be a number, got {params.get(key)!r}") from None
    if value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ModuleInitializationError(f"{key} must be {bound}, got {value}")
    return value


def parse_optional_zones(raw_zones: Sequence[Dict] | None) -> List[Zone]:
    """Zones from config, where *no* zones is valid and means the whole frame."""
    if not raw_zones:
        return []
    try:
        zones = [Zone.from_config(raw) for raw in raw_zones]
    except ZoneError as exc:
        raise ModuleInitializationError(str(exc)) from exc
    names = [zone.name for zone in zones]
    if len(set(names)) != len(names):
        raise ModuleInitializationError(f"Duplicate zone names: {sorted(names)}")
    return zones


def zone_containing(zones: Sequence[Zone], point: Tuple[float, float], context: FrameContext) -> str | None:
    """Name of the first zone holding ``point``; ``""`` when no zones are set
    (the whole frame counts); ``None`` when zones are set and none holds it."""
    if not zones:
        return ""
    for zone in zones:
        if zone.contains(point, context.width, context.height):
            return zone.name
    return None


def make_event(
    module_name: str,
    event_type: str,
    context: FrameContext,
    severity: Severity,
    subject: Dict[str, Any] | None = None,
    data: Dict[str, Any] | None = None,
) -> Event:
    return Event(
        event_type=event_type,
        source_module=module_name,
        frame_id=context.frame_id,
        timestamp=context.timestamp_iso,
        severity=severity,
        subject=dict(subject or {}),
        data={key: value for key, value in (data or {}).items() if value not in (None, "")},
    )


def gather_tracks(context: FrameContext, sources: Iterable[str], classes: Sequence[str] = ()) -> List[Any]:
    tracks: List[Any] = []
    for source in sources:
        for track in context.get_tracks(source):
            if classes and track.label not in classes:
                continue
            tracks.append(track)
    return tracks


class SubjectMemory(Generic[State]):
    """Per-track state that expires when a track stops being reported.

    A tracker only reports a track on frames where it matched a detection, so
    a subject can be missing for a frame or two and then return with the same
    id. State is therefore kept for ``forget_seconds`` after the last sighting
    rather than dropped the first frame the subject is absent.
    """

    def __init__(self, forget_seconds: float = 5.0) -> None:
        self.forget_seconds = forget_seconds
        self._states: Dict[SubjectKey, State] = {}
        self._last_seen: Dict[SubjectKey, float] = {}

    def get(self, key: SubjectKey) -> State | None:
        return self._states.get(key)

    def put(self, key: SubjectKey, state: State, now: float) -> State:
        self._states[key] = state
        self._last_seen[key] = now
        return state

    def drop(self, key: SubjectKey) -> None:
        self._states.pop(key, None)
        self._last_seen.pop(key, None)

    def touch(self, key: SubjectKey, now: float) -> None:
        self._last_seen[key] = now

    def expire(self, now: float) -> List[Tuple[SubjectKey, State]]:
        """Drop and return subjects unseen for longer than ``forget_seconds``."""
        gone = [key for key, seen in self._last_seen.items() if now - seen > self.forget_seconds]
        dropped = [(key, self._states[key]) for key in gone if key in self._states]
        for key in gone:
            self._states.pop(key, None)
            self._last_seen.pop(key, None)
        return dropped

    def items(self) -> List[Tuple[SubjectKey, State]]:
        return list(self._states.items())

    def last_seen(self, key: SubjectKey) -> float | None:
        return self._last_seen.get(key)

    def __len__(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        self._states.clear()
        self._last_seen.clear()
