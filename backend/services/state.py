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
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)


#: A stream reads several times a second; a thumbnail about once a second.
_STREAM_DEMAND_SEC = 1.5
_SNAPSHOT_DEMAND_SEC = 4.0


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
    _last_publish_at: Optional[float] = field(default=None, repr=False)
    _stream_read_at: Optional[float] = field(default=None, repr=False)
    _snapshot_read_at: Optional[float] = field(default=None, repr=False)

    #: Static description of the run, set once at startup.
    info: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    running: bool = True
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _cached_snapshot: Optional[Dict[str, Any]] = field(default=None, repr=False)

    #: Called with each batch of newly published events, outside the lock -
    #: how the alert center hears about them. Must not block for long.
    on_events: Optional[Callable[[List[Dict[str, Any]]], Any]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._frame_ready = threading.Condition(self._lock)
        self._events = deque(maxlen=self.max_events)
        self._cached_snapshot = self._build_snapshot()

    def _build_snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "running": self.running,
            "uptime_sec": round(time.time() - self.started_at, 1),
            "info": dict(self.info),
            "stats": dict(self._stats),
            "frame": dict(self._summary),
            "event_count": self._event_seq,
        }

    def update_info(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self.info = dict(info)
            if self._cached_snapshot is not None:
                self._cached_snapshot = {**self._cached_snapshot, "info": dict(info)}

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
            self._last_publish_at = time.monotonic()
            if jpeg is not None:
                self._jpeg = jpeg
                self._frame_seq += 1
            self._summary = summary
            self._stats = stats
            for event in events:
                self._event_seq += 1
                self._events.append({"id": self._event_seq, **event})
            self._cached_snapshot = {
                "running": self.running,
                "uptime_sec": round(time.time() - self.started_at, 1),
                "info": dict(self.info),
                "stats": dict(self._stats),
                "frame": dict(self._summary),
                "event_count": self._event_seq,
            }
            self._frame_ready.notify_all()
        if events and self.on_events is not None:
            try:
                self.on_events(events)
            except Exception:  # noqa: BLE001 - alerting must never stop the pipeline
                logger.exception("Alert handling failed; the events are still in the feed")

    def publish_frame(self, jpeg: bytes) -> None:
        """Replace only the picture, leaving results, stats and events alone.

        Used by the live view, which shows the camera at its own rate between
        analysed frames. It deliberately does not count as the pipeline
        publishing: a camera whose analysis has hung must still read as
        stalled, however smoothly its picture moves.
        """
        with self._frame_ready:
            self._jpeg = jpeg
            self._frame_seq += 1
            self._frame_ready.notify_all()

    def stop(self) -> None:
        with self._frame_ready:
            self.running = False
            if self._cached_snapshot is not None:
                self._cached_snapshot = {**self._cached_snapshot, "running": False}
            self._frame_ready.notify_all()

    def reset_session(self) -> str:
        """Reset the camera session ID and clear old frame/detections."""
        with self._frame_ready:
            self.session_id = uuid.uuid4().hex[:12]
            self._summary = {}
            self._jpeg = None
            self._cached_snapshot = self._build_snapshot()
            self._frame_ready.notify_all()
            return self.session_id

    def resume(self) -> None:
        """Flip ``running`` back on after a module reload rebuilds the pipeline.

        The pipeline's own shutdown always calls :meth:`stop` on its way out
        (closing the output sink), including the brief stop-and-rebuild a
        module change causes; this is what tells the dashboard the *next*
        pipeline run has started without tearing down the server itself.
        """
        with self._frame_ready:
            self.running = True
            self.session_id = uuid.uuid4().hex[:12]
            self._summary = {}
            self._cached_snapshot = self._build_snapshot()
            self._frame_ready.notify_all()

    # ------------------------------------------------------------------ #
    # Reading (server threads)
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Any]:
        """Completely non-blocking read from in-memory cached state dictionary."""
        cached = self._cached_snapshot
        if cached is not None:
            return {
                "session_id": cached.get("session_id", self.session_id),
                "running": cached["running"],
                "uptime_sec": round(time.time() - self.started_at, 1),
                "info": dict(cached["info"]),
                "stats": dict(cached["stats"]),
                "frame": dict(cached["frame"]),
                "event_count": cached["event_count"],
            }
        with self._lock:
            return self._build_snapshot()

    def seconds_since_publish(self) -> Optional[float]:
        """Seconds since the pipeline last published, or None before the first."""
        with self._lock:
            if self._last_publish_at is None:
                return None
            return time.monotonic() - self._last_publish_at

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            self._snapshot_read_at = time.monotonic()
            return self._jpeg

    def latest_frame(self) -> Tuple[Optional[bytes], int]:
        """``(jpeg, seq)`` of the newest frame, without waiting."""
        with self._lock:
            self._stream_read_at = time.monotonic()
            return self._jpeg, self._frame_seq

    def viewer_demand(self) -> str:
        """Who is looking at this camera's picture right now.

        ``"stream"`` - a live stream is open; ``"snapshot"`` - only stills are
        being fetched (a thumbnail); ``"idle"`` - nobody. Lets the live view
        spend JPEG encoding where someone will actually see it.
        """
        now = time.monotonic()
        with self._lock:
            stream, snapshot = self._stream_read_at, self._snapshot_read_at
        if stream is not None and now - stream < _STREAM_DEMAND_SEC:
            return "stream"
        if snapshot is not None and now - snapshot < _SNAPSHOT_DEMAND_SEC:
            return "snapshot"
        return "idle"

    @property
    def last_event_id(self) -> int:
        """Id of the newest event ever published (0 before the first)."""
        with self._lock:
            return self._event_seq

    def events_since(self, since: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Events with an id greater than ``since``, newest first.

        When more than ``limit`` are newer, the page holds the *oldest*
        ``limit`` of them, so a client that advances ``since`` to the highest
        id it received pages forward through a burst. Returning the newest
        page instead would make it skip the rest of the burst forever.
        """
        with self._lock:
            newer = [e for e in self._events if e["id"] > since]
        return list(reversed(newer[:limit]))

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
