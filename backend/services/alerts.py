"""Alerts: the events someone should look at, until someone has.

An event is a fact a module published ("person #4 in 'server room'"). An
alert is the dashboard's to-do item built from it: only events at or above
``min_severity`` become alerts, repeats of the same thing on the same camera
fold into the open alert instead of piling up, and an alert stays *open*
until a person acknowledges it.

This is the downstream layer Agent.md §16 keeps out of the modules: no module
knows alerts exist, and nothing here changes what a module does. It is
in-memory and per-run - it does not notify anyone outside the dashboard.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}

STATUS_OPEN = "open"
STATUS_ACKNOWLEDGED = "acknowledged"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class _Alert:
    id: int
    camera: str
    event_type: str
    module: str
    severity: str
    subject: Dict[str, Any]
    data: Dict[str, Any]
    first_seen: str
    last_seen: str
    last_seen_mono: float
    count: int = 1
    status: str = STATUS_OPEN
    acknowledged_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "camera": self.camera,
            "event": self.event_type,
            "module": self.module,
            "severity": self.severity,
            "subject": dict(self.subject),
            "data": dict(self.data),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "count": self.count,
            "status": self.status,
            "acknowledged_at": self.acknowledged_at,
        }


class AlertCenter:
    """Thread-safe, bounded store of alerts across every camera.

    ``ingest`` is called from pipeline threads; everything else from server
    threads. One lock guards all of it.
    """

    def __init__(
        self,
        min_severity: str = "warning",
        dedupe_seconds: float = 30.0,
        max_alerts: int = 500,
        clock=time.monotonic,
    ) -> None:
        if min_severity not in SEVERITY_RANK:
            raise ValueError(f"min_severity must be one of {list(SEVERITY_RANK)}, got {min_severity!r}")
        if dedupe_seconds < 0:
            raise ValueError("dedupe_seconds must be >= 0")
        if max_alerts < 1:
            raise ValueError("max_alerts must be >= 1")
        self.min_severity = min_severity
        self.dedupe_seconds = float(dedupe_seconds)
        self.max_alerts = int(max_alerts)
        self._clock = clock
        self._lock = threading.Lock()
        self._alerts: "OrderedDict[int, _Alert]" = OrderedDict()
        #: Fold key -> id of the open alert repeats fold into.
        self._open_by_key: Dict[Hashable, int] = {}
        self._seq = 0

    @classmethod
    def from_config(cls, config: Any) -> "AlertCenter":
        return cls(
            min_severity=config.min_severity,
            dedupe_seconds=config.dedupe_seconds,
            max_alerts=config.max_alerts,
        )

    # ------------------------------------------------------------------ #
    # Writing (pipeline threads)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fold_key(camera: str, event: Dict[str, Any]) -> Tuple:
        subject = event.get("subject") or {}
        zone = (event.get("data") or {}).get("zone")
        who = tuple(sorted((str(k), str(v)) for k, v in subject.items()))
        return (camera, event.get("module"), event.get("event"), who, zone)

    def ingest(self, camera: str, events: Iterable[Dict[str, Any]]) -> List[int]:
        """Turn qualifying events into alerts. Returns the ids touched."""
        touched: List[int] = []
        floor = SEVERITY_RANK[self.min_severity]
        with self._lock:
            for event in events:
                severity = str(event.get("severity", "info"))
                if SEVERITY_RANK.get(severity, 0) < floor:
                    continue
                now = self._clock()
                key = self._fold_key(camera, event)
                existing = self._alerts.get(self._open_by_key.get(key, -1))
                if (
                    existing is not None
                    and existing.status == STATUS_OPEN
                    and now - existing.last_seen_mono <= self.dedupe_seconds
                ):
                    existing.count += 1
                    existing.last_seen = event.get("timestamp") or _now_iso()
                    existing.last_seen_mono = now
                    existing.data = dict(event.get("data") or {})
                    if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(existing.severity, 0):
                        existing.severity = severity
                    touched.append(existing.id)
                    continue

                self._seq += 1
                seen = event.get("timestamp") or _now_iso()
                alert = _Alert(
                    id=self._seq,
                    camera=camera,
                    event_type=str(event.get("event", "")),
                    module=str(event.get("module", "")),
                    severity=severity,
                    subject=dict(event.get("subject") or {}),
                    data=dict(event.get("data") or {}),
                    first_seen=seen,
                    last_seen=seen,
                    last_seen_mono=now,
                )
                self._alerts[alert.id] = alert
                self._open_by_key[key] = alert.id
                touched.append(alert.id)
                self._trim()
        return touched

    def _trim(self) -> None:
        """Drop the oldest alerts beyond ``max_alerts``, acknowledged ones first."""
        while len(self._alerts) > self.max_alerts:
            victim = next(
                (a.id for a in self._alerts.values() if a.status == STATUS_ACKNOWLEDGED),
                next(iter(self._alerts)),
            )
            self._alerts.pop(victim)
            self._open_by_key = {k: v for k, v in self._open_by_key.items() if v != victim}

    # ------------------------------------------------------------------ #
    # Server threads
    # ------------------------------------------------------------------ #

    def acknowledge(self, alert_id: int) -> Optional[Dict[str, Any]]:
        """Mark one alert acknowledged. None when there is no such alert."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            if alert.status == STATUS_OPEN:
                alert.status = STATUS_ACKNOWLEDGED
                alert.acknowledged_at = _now_iso()
            return alert.to_dict()

    def acknowledge_all(self, camera: Optional[str] = None) -> int:
        """Acknowledge every open alert (on one camera, if given). Returns how many."""
        stamp = _now_iso()
        done = 0
        with self._lock:
            for alert in self._alerts.values():
                if alert.status == STATUS_OPEN and (camera is None or alert.camera == camera):
                    alert.status = STATUS_ACKNOWLEDGED
                    alert.acknowledged_at = stamp
                    done += 1
        return done

    def listing(
        self, *, status: str = "all", camera: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Newest-updated first, plus counts the header badge needs."""
        with self._lock:
            alerts = list(self._alerts.values())
        open_alerts = [a for a in alerts if a.status == STATUS_OPEN]
        chosen = alerts
        if status != "all":
            chosen = [a for a in chosen if a.status == status]
        if camera is not None:
            chosen = [a for a in chosen if a.camera == camera]
        chosen.sort(key=lambda a: (a.last_seen_mono, a.id), reverse=True)
        return {
            "alerts": [a.to_dict() for a in chosen[:limit]],
            "open_count": len(open_alerts),
            "open_critical": sum(1 for a in open_alerts if a.severity == "critical"),
            "total": len(alerts),
            "min_severity": self.min_severity,
        }
