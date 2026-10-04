"""The set of cameras a run is driving, and what each one is doing.

One :class:`~backend.services.state.DashboardState` per camera, so each camera
keeps its own latest frame, event history and event ids - the single-camera
dashboard code is reused unchanged, once per camera. The hub only adds the
bookkeeping around them: which cameras exist, in what order, and whether each
one is starting, live, stalled or gone.

States survive a reload (same id -> same state), so a module change does not
wipe a camera's event history, and a browser's event cursor stays valid.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from backend.services.state import DashboardState

#: A camera that has published nothing for this long is reported as stalled.
STALL_AFTER_SEC = 5.0

PHASE_STARTING = "starting"
PHASE_RUNNING = "running"
PHASE_ENDED = "ended"
PHASE_FAILED = "failed"


@dataclass
class _Slot:
    id: str
    name: str
    source_type: str
    state: DashboardState
    phase: str = PHASE_STARTING
    error: Optional[str] = None


class CameraHub:
    """Thread-safe registry of per-camera dashboard state."""

    def __init__(self, max_events: int = 200) -> None:
        self._max_events = max_events
        self._lock = threading.Lock()
        self._slots: Dict[str, _Slot] = {}

    # ------------------------------------------------------------------ #
    # Configuration (main thread)
    # ------------------------------------------------------------------ #

    def configure(self, cameras: Sequence[tuple[str, str, str]]) -> None:
        """Make the hub describe exactly ``cameras`` = ``(id, name, source_type)``, in order."""
        with self._lock:
            slots: Dict[str, _Slot] = {}
            for cam_id, name, source_type in cameras:
                slot = self._slots.get(cam_id)
                if slot is None:
                    slot = _Slot(cam_id, name, source_type, DashboardState(max_events=self._max_events))
                else:
                    slot.name, slot.source_type = name, source_type
                slot.phase, slot.error = PHASE_STARTING, None
                slots[cam_id] = slot
            self._slots = slots

    def set_phase(self, cam_id: str, phase: str, error: str | None = None) -> None:
        with self._lock:
            slot = self._slots.get(cam_id)
            if slot is not None:
                slot.phase, slot.error = phase, error

    def resume_all(self) -> None:
        """Mark every camera's stream as running again after a rebuild."""
        for state in self.states():
            state.resume()

    def stop_all(self) -> None:
        for state in self.states():
            state.stop()

    # ------------------------------------------------------------------ #
    # Reading (server threads)
    # ------------------------------------------------------------------ #

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._slots)

    def states(self) -> List[DashboardState]:
        with self._lock:
            return [slot.state for slot in self._slots.values()]

    @property
    def default_id(self) -> Optional[str]:
        with self._lock:
            return next(iter(self._slots), None)

    def get(self, cam_id: Optional[str]) -> Optional[DashboardState]:
        """The state for ``cam_id``; the first camera's when ``cam_id`` is None."""
        with self._lock:
            if cam_id is None:
                slot = next(iter(self._slots.values()), None)
            else:
                slot = self._slots.get(cam_id)
            return slot.state if slot is not None else None

    def describe(self) -> List[Dict[str, Any]]:
        """One dict per camera for ``GET /api/cameras`` - no URLs, no secrets."""
        with self._lock:
            slots = list(self._slots.values())

        rows: List[Dict[str, Any]] = []
        for slot in slots:
            snap = slot.state.snapshot()
            idle = slot.state.seconds_since_publish()
            if slot.phase == PHASE_FAILED:
                status = "failed"
            elif slot.phase == PHASE_ENDED:
                status = "offline"
            elif idle is None:
                status = "starting"
            elif idle > STALL_AFTER_SEC:
                status = "stalled"
            else:
                status = "live"
            frame = snap.get("frame") or {}
            stats = snap.get("stats") or {}
            rows.append(
                {
                    "id": slot.id,
                    "name": slot.name,
                    "source": slot.source_type,
                    "status": status,
                    "error": slot.error,
                    "fps": stats.get("fps"),
                    "resolution": frame.get("resolution"),
                    "event_count": snap.get("event_count", 0),
                }
            )
        return rows
