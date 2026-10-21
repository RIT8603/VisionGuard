"""MVP features: event log, gate entry/exit, liveness, unknown snapshots, reports API."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.persistence.event_log import EventLog
from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.inference.face_base import RawFace
from computer_vision.modules.analysis.gate import GateCounterModule
from computer_vision.modules.recognition.face_liveness import (
    FaceLivenessModule,
    similarity_residual,
)


def ctx(frame_id, t=None, frame=None):
    return FrameContext(
        frame=frame if frame is not None else np.zeros((400, 600, 3), np.uint8),
        frame_id=frame_id, source_id="cam-1", monotonic_ts=float(frame_id if t is None else t),
    )


# ------------------------------------------------------------------ event log


def test_event_log_stores_queries_and_exports(tmp_path):
    log = EventLog(tmp_path / "events.db")
    log.add_many("cam-1", [
        {"event": "gate_entry", "module": "gate_counter", "timestamp": "2026-10-20T08:00:00+00:00",
         "severity": "info", "subject": {"track_id": 4}, "data": {"direction": "in", "identity": "Asha"}},
        {"event": "unknown_person", "timestamp": "2026-10-20T08:01:00+00:00", "severity": "warning",
         "data": {"snapshot": "x.jpg"}},
    ])
    everything = log.query()
    assert everything["total"] == 2 and everything["events"][0]["event"] == "unknown_person"
    gate = log.query(events=["gate_entry"])
    assert gate["events"][0]["identity"] == "Asha" and gate["events"][0]["track_id"] == 4
    csv_text = log.to_csv(events=["gate_entry"])
    assert "gate_entry" in csv_text and "Asha" in csv_text and ",in," in csv_text


# ---------------------------------------------------------------- gate counter


def gate(**params):
    module = GateCounterModule({"line": [[0.0, 0.5], [1.0, 0.5]], "inside": "right", **params})
    module.initialize()
    return module


def walker(c, track_id, foot_y, name=None):
    c.add_tracks("person_tracking", [Track(track_id, "person", BBox(280, foot_y - 120, 320, foot_y),
                                           source_module="person_tracking")])
    if name:
        c.add_detections("face_recognition", [Detection(
            name, 0.9, BBox(290, foot_y - 120, 310, foot_y - 100), source_module="face_recognition",
            extra={"identity": name, "recognised": True, "person_track_id": track_id})])


def test_crossing_down_is_an_entry_and_back_up_an_exit():
    module = gate()
    events = []
    # line a->b runs left to right at y=200; "right" of it in image coords is below.
    for i, y in enumerate([150, 170, 230, 260, 230, 170, 150]):
        c = ctx(i)
        walker(c, 1, y, name="Asha" if i == 0 else None)
        module.process(c)
        events += [e.event_type for e in c.events]
        if i == 3:
            assert module.inside == 1
    assert events == ["gate_entry", "gate_exit"]
    assert module.inside == 0


def test_the_entry_names_the_person_recognised_earlier():
    module = gate()
    named = []
    for i, y in enumerate([120, 160, 240]):
        c = ctx(i)
        walker(c, 7, y, name="Ravi" if i == 0 else None)
        module.process(c)
        named += [e.data["identity"] for e in c.events]
    assert named == ["Ravi"]


def test_hovering_on_the_line_does_not_count():
    module = gate()
    for i, y in enumerate([150, 198, 202, 199, 201, 150]):
        c = ctx(i)
        walker(c, 2, y)
        module.process(c)
        assert not c.events


def test_inside_side_is_configurable():
    module = gate(inside="left")
    out = []
    for i, y in enumerate([150, 260]):
        c = ctx(i)
        walker(c, 3, y)
        module.process(c)
        out += [e.event_type for e in c.events]
    assert out == ["gate_exit"]


# -------------------------------------------------------------------- liveness

LM = np.array([[150, 170], [250, 170], [200, 223], [160, 270], [240, 270]], np.float32)


def test_a_flat_photo_moves_rigidly_and_a_head_does_not():
    angle = np.radians(10)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    photo = (LM - LM.mean(0)) @ rot.T * 1.2 + LM.mean(0) + [30, -10]
    assert similarity_residual(LM, photo) < 1e-3
    head = LM.copy()
    head[2, 0] += 20  # the nose swings when the head turns
    assert similarity_residual(LM, head) > 0.06


def face_frame(i, nose_dx=0.0, rng=np.random.default_rng(0)):
    frame = np.random.default_rng(1).integers(40, 200, (400, 600, 3), dtype=np.uint8)
    lm = LM.copy()
    lm[2, 0] += nose_dx
    c = ctx(i, t=i / 5.0, frame=frame)
    raw = RawFace(0.95, (120.0, 120.0, 280.0, 300.0), lm)
    c.add_detections("face_detection", [Detection("face", 0.95, BBox(120, 120, 280, 300),
                                                  source_module="face_detection", extra={"face": raw})])
    return c


def run_liveness(nose_moves):
    module = FaceLivenessModule({"decision_seconds": 4, "block_spoof": True})
    module.initialize()
    last = None
    events = []
    for i, dx in enumerate(nose_moves):
        c = face_frame(i, dx)
        module.process(c)
        last = c.get_detections("face_detection")[0].extra["liveness"]["state"]
        events += [e.event_type for e in c.events]
    return last, events


def test_a_still_photo_is_judged_a_spoof_after_the_decision_time():
    state, events = run_liveness([0.0] * 30)
    assert state == "spoof" and events == ["spoof_suspected"]


def test_a_head_that_moves_in_3d_is_live():
    moves = [0, 12, -10, 14, -8, 10, 0, 0, 0, 0] * 3
    state, events = run_liveness(moves)
    assert state == "live" and events == []


def test_a_face_is_checking_before_the_decision_time():
    state, _ = run_liveness([0.0] * 6)
    assert state == "checking"


def test_recognition_refuses_to_name_a_spoof():
    from computer_vision.identity.shared import get_shared_store, replace_shared_store
    from computer_vision.modules.recognition.face_recognition import FaceRecognitionModule

    replace_shared_store([("Asha", [np.ones(128, np.float32)])])
    module = FaceRecognitionModule({"min_quality": 0, "min_sharpness": 0, "min_face_size": 0})
    module._embedder = SimpleNamespace(embed=lambda frame, face: np.ones(128, np.float32))
    module._store = get_shared_store()
    c = face_frame(0)
    c.get_detections("face_detection")[0].extra["liveness"] = {"state": "spoof", "block": True}
    module.process(c)
    [det] = c.get_detections("face_recognition")
    assert det.label == "spoof" and det.extra["recognised"] is False
    replace_shared_store([])


# ------------------------------------------------------------ unknown snapshot


def test_unknown_person_alert_saves_a_snapshot(tmp_path):
    from computer_vision.modules.recognition.unknown_person import UnknownPersonModule

    module = UnknownPersonModule({"min_sightings": 1, "snapshot_dir": str(tmp_path)})
    c = ctx(5, frame=np.full((400, 600, 3), 120, np.uint8))
    c.add_detections("face_recognition", [Detection(
        "unknown", 0.9, BBox(100, 100, 200, 220), source_module="face_recognition",
        extra={"recognised": False, "verified": True, "similarity": 0.2})])
    module.process(c)
    [event] = c.events
    assert (tmp_path / event.data["snapshot"]).is_file()
    assert (tmp_path / event.data["frame_snapshot"]).is_file()


# ----------------------------------------------------------------- reports API

fastapi = pytest.importorskip("fastapi", reason="dashboard extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.server import build_app  # noqa: E402
from backend.services.hub import CameraHub  # noqa: E402


@pytest.fixture
def client(tmp_path):
    from backend.api.reports import day_bounds

    hub = CameraHub(event_log=EventLog(tmp_path / "events.db"))
    hub.configure([("cam-1", "Gate", "file")])
    since, _ = day_bounds(None)
    t0 = since.replace("+00:00", "")
    hub.event_log.add_many("cam-1", [
        {"event": "gate_entry", "timestamp": f"{t0}1", "data": {"direction": "in", "identity": "Asha"}},
        {"event": "gate_entry", "timestamp": f"{t0}2", "data": {"direction": "in", "identity": "Ravi"}},
        {"event": "gate_exit", "timestamp": f"{t0}3", "data": {"direction": "out", "identity": "Ravi"}},
        {"event": "gate_entry", "timestamp": f"{t0}4", "data": {"direction": "in", "identity": "unknown"}},
    ])
    (tmp_path / "snaps").mkdir()
    (tmp_path / "snaps" / "unknown_cam-1_x.jpg").write_bytes(b"\xff\xd8jpeg")
    app = build_app(hub.get("cam-1"), hub=hub, snapshot_dir=str(tmp_path / "snaps"))
    return TestClient(app, base_url="http://127.0.0.1")


def test_live_report_counts_who_is_inside(client):
    body = client.get("/api/reports/live").json()
    today = body["today"]
    assert today["entered"] == 3 and today["exited"] == 1 and today["inside"] == 2
    assert [p["identity"] for p in today["people_inside"]] == ["Asha"]
    assert body["cameras"][0]["id"] == "cam-1"


def test_event_log_filters_and_csv(client):
    exits = client.get("/api/reports/events", params={"event": "gate_exit"}).json()
    assert exits["total"] == 1 and exits["events"][0]["identity"] == "Ravi"
    csv_response = client.get("/api/reports/events.csv", params={"event": "gate_entry,gate_exit"})
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert csv_response.text.count("\n") == 5  # header + 4 rows
    attendance = client.get("/api/reports/attendance.csv").text
    assert "Asha" in attendance and "Ravi" in attendance and "unknown" not in attendance


def test_snapshots_are_served_but_paths_are_not(client):
    assert client.get("/api/snapshots/unknown_cam-1_x.jpg").status_code == 200
    assert client.get("/api/snapshots/..%2Fevents.db").status_code == 404
    assert client.get("/api/snapshots/missing.jpg").status_code == 404
