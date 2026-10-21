"""HTTP routes for the Reports page: live counts, the event log, CSV export, snapshots.

* ``GET /api/reports/live`` - per camera: people in view, who is recognised,
  unknown faces; plus today's gate totals (entered / exited / inside) and who
  is inside now, from the durable event log so a restart does not reset them.
* ``GET /api/reports/events`` - the event log, filtered and paged.
* ``GET /api/reports/events.csv`` - the same filters, as a CSV download.
* ``GET /api/reports/attendance.csv`` - one row per person for a day: first
  entry, last exit, number of entries and exits.
* ``GET /api/snapshots/{name}`` - an unknown-person snapshot image.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.hub import CameraHub
from computer_vision.utils.logger import get_logger

try:
    # Module level: FastAPI resolves these hints through this module's globals.
    from fastapi import Query
    from fastapi.responses import FileResponse, JSONResponse, Response
except ImportError:  # pragma: no cover - dashboard dependencies missing
    Query = FileResponse = JSONResponse = Response = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)

GATE_EVENTS = ("gate_entry", "gate_exit")
_SNAPSHOT_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,160}\.jpg$")


def day_bounds(day: Optional[str]) -> tuple[str, str]:
    """UTC ISO bounds of a local calendar day (``YYYY-MM-DD``; today when None)."""
    target = date.fromisoformat(day) if day else date.today()
    start = datetime.combine(target, dtime.min).astimezone().astimezone(timezone.utc)
    end = datetime.combine(target, dtime.max).astimezone().astimezone(timezone.utc)
    return start.isoformat(), end.isoformat()


def _split(value: Optional[str]) -> List[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def gate_presence(log: Any, since: str, until: str) -> Dict[str, Any]:
    """Entries, exits and who is inside, from the day's gate events."""
    page = log.query(events=GATE_EVENTS, since=since, until=until, limit=1000)
    events = list(reversed(page["events"]))  # oldest first
    entered = sum(1 for e in events if e["event"] == "gate_entry")
    exited = len(events) - entered
    last: Dict[str, Dict[str, Any]] = {}
    for e in events:
        name = e["identity"]
        if name and name != "unknown":
            last[name] = e
    inside_people = sorted(
        (
            {"identity": name, "since": e["timestamp"], "camera": e["camera"]}
            for name, e in last.items()
            if e["event"] == "gate_entry"
        ),
        key=lambda row: row["since"],
    )
    return {
        "entered": entered,
        "exited": exited,
        "inside": max(0, entered - exited),
        "people_inside": inside_people,
        "truncated": page["total"] > len(events),
    }


def attendance_rows(log: Any, since: str, until: str) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    offset = 0
    while True:
        page = log.query(events=GATE_EVENTS, since=since, until=until, limit=1000, offset=offset)["events"]
        for e in page:
            name = e["identity"] or "unknown"
            if name == "unknown":
                continue
            row = rows.setdefault(name, {"identity": name, "first_entry": "", "last_exit": "",
                                         "entries": 0, "exits": 0})
            if e["event"] == "gate_entry":
                row["entries"] += 1
                if not row["first_entry"] or e["timestamp"] < row["first_entry"]:
                    row["first_entry"] = e["timestamp"]
            else:
                row["exits"] += 1
                if e["timestamp"] > row["last_exit"]:
                    row["last_exit"] = e["timestamp"]
        if len(page) < 1000:
            break
        offset += len(page)
    return sorted(rows.values(), key=lambda r: r["identity"].lower())


def register_report_routes(app, hub: Optional[CameraHub], snapshot_dir: str | Path) -> None:
    snapshots = Path(snapshot_dir)

    def log_or_error():
        log = getattr(hub, "event_log", None) if hub is not None else None
        if log is None:
            return None, JSONResponse({"error": "The event log is not available in this run."}, 503)
        return log, None

    @app.get("/api/reports/live")
    def reports_live():
        cameras: List[Dict[str, Any]] = []
        if hub is not None:
            for row in hub.describe():
                state = hub.get(row["id"])
                frame = (state.snapshot().get("frame") if state is not None else None) or {}
                tracks = (frame.get("tracks") or {}).get("person_tracking") or []
                faces = (frame.get("detections") or {}).get("face_recognition") or []
                recognised = sorted({
                    f.get("extra", {}).get("identity") for f in faces
                    if f.get("extra", {}).get("recognised")
                } - {None})
                cameras.append({
                    "id": row["id"],
                    "name": row["name"],
                    "status": row["status"],
                    "people_in_view": len(tracks),
                    "recognised": recognised,
                    "unknown_faces": sum(1 for f in faces if f.get("class") == "unknown"),
                    "spoof_faces": sum(1 for f in faces if f.get("class") == "spoof"),
                })
        body: Dict[str, Any] = {
            "cameras": cameras,
            "people_in_view": sum(c["people_in_view"] for c in cameras),
        }
        log = getattr(hub, "event_log", None) if hub is not None else None
        if log is not None:
            since, until = day_bounds(None)
            body["today"] = gate_presence(log, since, until)
            body["today"]["by_type"] = log.summary(since)["by_type"]
        return JSONResponse(body)

    def _filters(camera, event, identity, day, since, until) -> Dict[str, Any]:
        if day:
            since, until = day_bounds(day)
        return {
            "camera": camera or None,
            "events": _split(event),
            "identity": identity or None,
            "since": since or None,
            "until": until or None,
        }

    @app.get("/api/reports/events")
    def reports_events(
        camera: Optional[str] = Query(None, max_length=64),
        event: Optional[str] = Query(None, max_length=400),
        identity: Optional[str] = Query(None, max_length=80),
        day: Optional[str] = Query(None, max_length=10),
        since: Optional[str] = Query(None, max_length=40),
        until: Optional[str] = Query(None, max_length=40),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        log, error = log_or_error()
        if error is not None:
            return error
        try:
            filters = _filters(camera, event, identity, day, since, until)
        except ValueError:
            return JSONResponse({"error": "day must be YYYY-MM-DD"}, 400)
        return JSONResponse(log.query(**filters, limit=limit, offset=offset))

    @app.get("/api/reports/events.csv", include_in_schema=False)
    def reports_events_csv(
        camera: Optional[str] = Query(None, max_length=64),
        event: Optional[str] = Query(None, max_length=400),
        identity: Optional[str] = Query(None, max_length=80),
        day: Optional[str] = Query(None, max_length=10),
        since: Optional[str] = Query(None, max_length=40),
        until: Optional[str] = Query(None, max_length=40),
    ):
        log, error = log_or_error()
        if error is not None:
            return error
        try:
            filters = _filters(camera, event, identity, day, since, until)
        except ValueError:
            return JSONResponse({"error": "day must be YYYY-MM-DD"}, 400)
        name = f"visionguard-events-{day or date.today().isoformat()}.csv"
        return Response(
            content=log.to_csv(**filters),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.get("/api/reports/attendance.csv", include_in_schema=False)
    def reports_attendance_csv(day: Optional[str] = Query(None, max_length=10)):
        log, error = log_or_error()
        if error is not None:
            return error
        try:
            since, until = day_bounds(day)
        except ValueError:
            return JSONResponse({"error": "day must be YYYY-MM-DD"}, 400)
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["identity", "first_entry", "last_exit", "entries", "exits"])
        writer.writeheader()
        writer.writerows(attendance_rows(log, since, until))
        name = f"visionguard-attendance-{day or date.today().isoformat()}.csv"
        return Response(
            content=out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.get("/api/snapshots/{name}", include_in_schema=False)
    def snapshot_file(name: str):
        # A bare file name only: no directories, no traversal.
        if not _SNAPSHOT_NAME.match(name):
            return Response(status_code=404)
        path = snapshots / name
        if not path.is_file():
            return Response(status_code=404)
        return FileResponse(path, media_type="image/jpeg")
