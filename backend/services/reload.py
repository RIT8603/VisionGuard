"""Hands a change request from the dashboard API thread to the main loop.

``POST /api/modules`` and ``POST /api/cameras`` run on the dashboard's server
thread and cannot touch the pipelines directly - they only know what the
operator asked for. They call :meth:`ReloadController.request` (a module list)
or :meth:`ReloadController.request_cameras` (a camera list); the loop in
``backend.main`` polls :meth:`pop` / :meth:`pop_cameras` once its current
pipelines have stopped and rebuilds from the new values. A single pending value
of each kind is enough: a second request before the first is applied simply
replaces it, and the operator sees the latest selection win.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence


class ReloadController:
    """Thread-safe hand-off of desired module / camera lists between two threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trigger = threading.Event()
        self._pending: Optional[List[str]] = None
        self._pending_cameras: Optional[List[Any]] = None
        self._pending_settings: Optional[Dict[str, Dict[str, Any]]] = None
        #: Set by the main loop after a failed attempt; read back by the API
        #: so a rejected selection can be surfaced instead of disappearing.
        self.last_error: Optional[str] = None

    def request(self, enabled: List[str]) -> None:
        """Queue ``enabled`` as the desired module set and wake the run loop."""
        with self._lock:
            self._pending = list(enabled)
            self.last_error = None
        self._trigger.set()

    def request_cameras(self, cameras: Sequence[Any]) -> None:
        """Queue ``cameras`` (validated ``CameraConfig`` objects) as the new camera set."""
        with self._lock:
            self._pending_cameras = list(cameras)
            self.last_error = None
        self._trigger.set()

    def request_settings(self, module: str, settings: Dict[str, Any]) -> None:
        """Queue new values for some of ``module``'s settings.

        Unlike the lists above these accumulate: two changes to different
        modules (or keys) before the loop applies them are both kept.
        """
        with self._lock:
            pending = self._pending_settings or {}
            pending.setdefault(module, {}).update(settings)
            self._pending_settings = pending
            self.last_error = None
        self._trigger.set()

    def wait(self) -> None:
        """Block until a request arrives.

        Call this from a fresh daemon thread for each pipeline run - once it
        returns, that thread's only job (telling the pipelines to stop) is
        done and it exits.
        """
        self._trigger.wait()

    def wait_for(self, timeout: float) -> bool:
        """Like :meth:`wait`, but gives up after ``timeout`` seconds."""
        return self._trigger.wait(timeout)

    @property
    def pending(self) -> bool:
        """True while a request has been made and not yet consumed."""
        return self._trigger.is_set()

    def pop(self) -> Optional[List[str]]:
        """Consume and clear the pending module request, if any."""
        with self._lock:
            pending, self._pending = self._pending, None
            self._clear_if_idle()
        return pending

    def pop_cameras(self) -> Optional[List[Any]]:
        """Consume and clear the pending camera request, if any."""
        with self._lock:
            pending, self._pending_cameras = self._pending_cameras, None
            self._clear_if_idle()
        return pending

    def pop_settings(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Consume and clear the pending settings request, if any."""
        with self._lock:
            pending, self._pending_settings = self._pending_settings, None
            self._clear_if_idle()
        return pending

    def _clear_if_idle(self) -> None:
        # Caller holds the lock. The wake-up flag stays up while the other
        # kind of request is still waiting to be consumed.
        if (
            self._pending is None
            and self._pending_cameras is None
            and self._pending_settings is None
        ):
            self._trigger.clear()

    def fail(self, message: str) -> None:
        """Record why the last requested change could not be applied."""
        with self._lock:
            self.last_error = message
