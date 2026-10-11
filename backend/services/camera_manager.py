"""What the dashboard may ask about, and ask for, regarding cameras.

Sits between the HTTP layer (which only routes requests) and the main loop
(which owns the pipelines): ``GET /api/cameras`` reads from here, and
``POST /api/cameras`` validates here and then hands the new list to the main
loop through the :class:`~backend.services.reload.ReloadController`. Like a
module change, accepting a camera change only means it was queued.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.services.cameras import parse_cameras, public_view
from backend.services.hub import CameraHub
from backend.services.reload import ReloadController
from computer_vision.core.config import CameraConfig
from computer_vision.modules.registry import ModuleRegistry


class CameraManager:
    """Thread-safe view of the running camera list plus the one way to change it."""

    def __init__(
        self,
        hub: CameraHub,
        reload: ReloadController | None,
        *,
        max_cameras: int,
        editable: bool,
    ) -> None:
        self._hub = hub
        self._reload = reload
        self._max = max_cameras
        self._editable = editable
        self._lock = threading.Lock()
        self._current: List[CameraConfig] = []
        self._details: Dict[str, Dict[str, Any]] = {}
        self._shared_modules: List[str] | None = None

    @property
    def max_cameras(self) -> int:
        return self._max

    @property
    def editable(self) -> bool:
        return self._editable and self._reload is not None

    def set_current(
        self,
        cameras: Sequence[CameraConfig],
        details: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        shared_modules: Sequence[str] | None = None,
    ) -> None:
        """Record the camera list now running (main loop, after every rebuild).

        ``details`` can supply the public view for a camera that cannot be
        expressed as a :class:`CameraConfig` (a ``video:`` source with an
        unusual URL scheme), keyed by id. ``shared_modules`` is the module
        selection every camera without its own ``modules`` list follows.
        """
        with self._lock:
            self._current = list(cameras)
            self._details = {k: dict(v) for k, v in (details or {}).items()}
            self._shared_modules = None if shared_modules is None else list(shared_modules)

    def current(self) -> List[CameraConfig]:
        with self._lock:
            return list(self._current)

    def listing(self) -> Dict[str, Any]:
        """Body of ``GET /api/cameras``."""
        with self._lock:
            by_id = {cam.id: public_view(cam) for cam in self._current}
            by_id.update({k: dict(v) for k, v in self._details.items()})
            shared = None if self._shared_modules is None else list(self._shared_modules)
        rows = []
        for row in self._hub.describe():
            merged = {**by_id.get(row["id"], {}), **row}
            rows.append(merged)
        return {
            "cameras": rows,
            "max_cameras": self._max,
            "editable": self.editable,
            "shared_modules": shared,
            "apply_error": self._reload.last_error if self._reload is not None else None,
        }

    def submit(self, items: Any) -> Tuple[int, Dict[str, Any]]:
        """Validate ``items`` and queue them. Returns ``(http_status, body)``."""
        if not self._editable:
            return 403, {"error": "Camera editing is turned off (dashboard.allow_camera_edit)."}
        if self._reload is None:
            return 503, {"error": "This run was started without --dashboard camera support."}
        if not isinstance(items, list):
            return 400, {"error": "'cameras' must be a list of camera objects"}

        cameras, errors = parse_cameras(
            items,
            max_cameras=self._max,
            existing=self.current(),
            known_modules=ModuleRegistry.known_names(),
        )
        if errors:
            return 400, {"error": "; ".join(errors)}

        self._reload.request_cameras(cameras)
        return 202, {"status": "applying", "cameras": [public_view(cam) for cam in cameras]}
