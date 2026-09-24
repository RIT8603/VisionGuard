"""Shared state between the pipeline thread and the HTTP server thread.

The pipeline writes; request handlers read. Everything crossing that boundary
goes through one lock, and readers always get a complete snapshot rather than
a half-updated one.

Only the *latest* frame is kept. A dashboard that queues frames would drift
further behind the longer it ran, which is the same mistake the RTSP source
avoids - freshness beats completeness for a live view.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class DashboardState:
    """Thread-safe snapshot of what the pipeline is currently doing."""

    #: How many recent events to keep for the events feed.
    max_events: int = 200

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _frame_ready: threading.Condition = field(init=False, repr=False)

    _jpeg: Optional[bytes] = field(default=None, repr=False)
    _frame_seq: int = 0
    _summary: Dict[str, Any] = field(default_factory=dict, repr=False)
    _stats: Dict[str, Any] = field(default_factory=dict, repr=False)
    _events: Deque[Dict[str, Any]] = field(default_factory=deque, repr=False)
    _event_seq: int = 0

    #: Static description of the run, set once at startup.
    info: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    running: bool = True

    def __post_init__(self) -> None:
        self._frame_ready = threading.Condition(self._lock)
        self._events = deque(maxlen=self.max_events)

    # ------------------------------------------------------------------ #
    # Writing (pipeline thread)
    # ------------------------------------------------------------------ #

    def publish(
        self,
        jpeg: Optional[bytes],
        summary: Dict[str, Any],
        stats: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> None:
        """Replace the current snapshot and wake any waiting stream readers."""
        with self._frame_ready:
            if jpeg is not None:
                self._jpeg = jpeg
                self._frame_seq += 1
            self._summary = summary
            self._stats = stats
            for event in events:
                self._event_seq += 1
                self._events.append({"id": self._event_seq, **event})
            self._frame_ready.notify_all()

    def stop(self) -> None:
        with self._frame_ready:
            self.running = False
            self._frame_ready.notify_all()

    # ------------------------------------------------------------------ #
    # Reading (server threads)
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "uptime_sec": round(time.time() - self.started_at, 1),
                "info": dict(self.info),
                "stats": dict(self._stats),
                "frame": dict(self._summary),
                "event_count": self._event_seq,
            }

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def events_since(self, since: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Newest-first events with an id greater than ``since``."""
        with self._lock:
            newer = [e for e in self._events if e["id"] > since]
        return list(reversed(newer))[:limit]

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than ``last_seq`` exists.

        Returns ``(jpeg, seq)``, or ``(None, last_seq)`` on timeout so the
        caller can re-check whether the pipeline is still running.
        """
        deadline = time.perf_counter() + timeout
        with self._frame_ready:
            while self._frame_seq <= last_seq and self.running:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None, last_seq
                self._frame_ready.wait(remaining)
            return self._jpeg, self._frame_seq
