"""Durable event log: every event from every camera, kept in SQLite.

The dashboard's live event feed holds the last 200 events in memory and loses
them on restart. Entry/exit records, unknown-person alerts and attendance need
to survive that and be exportable, so every event that reaches the dashboard is
also appended here (in the main process, whatever process the camera runs in).

Queried by the Reports page (filters, paging) and exported as CSV.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    camera    TEXT NOT NULL,
    ts        TEXT NOT NULL,
    event     TEXT NOT NULL,
    module    TEXT NOT NULL DEFAULT '',
    severity  TEXT NOT NULL DEFAULT 'info',
    track_id  INTEGER,
    identity  TEXT,
    subject   TEXT NOT NULL DEFAULT '{}',
    data      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_event ON events(event);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);
"""

CSV_COLUMNS = ("id", "time", "camera", "event", "severity", "track_id", "identity", "direction", "details")


def _identity_of(event: Dict[str, Any]) -> Optional[str]:
    for source in (event.get("data") or {}, event.get("subject") or {}):
        for key in ("identity", "name"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return None


class EventLog:
    """Append-only event store with simple queries."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #

    def add_many(self, camera: str, events: Iterable[Dict[str, Any]]) -> int:
        rows = []
        for event in events:
            subject = event.get("subject") or {}
            track = subject.get("track_id")
            rows.append((
                camera,
                str(event.get("timestamp") or ""),
                str(event.get("event") or event.get("event_type") or "event"),
                str(event.get("module") or event.get("source_module") or ""),
                str(event.get("severity") or "info"),
                int(track) if isinstance(track, (int, float)) else None,
                _identity_of(event),
                json.dumps(subject, default=str),
                json.dumps(event.get("data") or {}, default=str),
            ))
        if not rows:
            return 0
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO events (camera, ts, event, module, severity, track_id, identity, subject, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def _where(
        self,
        camera: Optional[str],
        events: Sequence[str],
        since: Optional[str],
        until: Optional[str],
        identity: Optional[str],
    ):
        clauses, args = [], []
        if camera:
            clauses.append("camera = ?")
            args.append(camera)
        if events:
            clauses.append(f"event IN ({','.join('?' for _ in events)})")
            args.extend(events)
        if since:
            clauses.append("ts >= ?")
            args.append(since)
        if until:
            clauses.append("ts <= ?")
            args.append(until)
        if identity:
            clauses.append("identity = ?")
            args.append(identity)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    def query(
        self,
        *,
        camera: Optional[str] = None,
        events: Sequence[str] = (),
        since: Optional[str] = None,
        until: Optional[str] = None,
        identity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        where, args = self._where(camera, events, since, until, identity)
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM events{where}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM events{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*args, max(1, min(int(limit), 1000)), max(0, int(offset))],
            ).fetchall()
        return {"total": int(total), "events": [self._row(r) for r in rows]}

    def summary(self, since: Optional[str] = None) -> Dict[str, Any]:
        """Counts by event type, and gate totals, since ``since`` (ISO time)."""
        where, args = self._where(None, (), since, None, None)
        with self._connect() as conn:
            by_type = {
                r["event"]: int(r["n"])
                for r in conn.execute(f"SELECT event, COUNT(*) AS n FROM events{where} GROUP BY event", args)
            }
        return {"by_type": by_type}

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "camera": row["camera"],
            "timestamp": row["ts"],
            "event": row["event"],
            "module": row["module"],
            "severity": row["severity"],
            "track_id": row["track_id"],
            "identity": row["identity"],
            "subject": json.loads(row["subject"] or "{}"),
            "data": json.loads(row["data"] or "{}"),
        }

    def to_csv(self, **filters: Any) -> str:
        """Every matching event (newest first) as CSV text."""
        filters.setdefault("limit", 1000)
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(CSV_COLUMNS)
        offset = 0
        while True:
            page = self.query(**{**filters, "offset": offset})["events"]
            for e in page:
                data = dict(e["data"])
                direction = data.pop("direction", "")
                details = "; ".join(f"{k}={v}" for k, v in data.items() if k not in ("identity",))
                writer.writerow([
                    e["id"], e["timestamp"], e["camera"], e["event"], e["severity"],
                    "" if e["track_id"] is None else e["track_id"], e["identity"] or "",
                    direction, details,
                ])
            if len(page) < filters["limit"]:
                break
            offset += len(page)
        return out.getvalue()
