"""Alert center and the /api/alerts endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.server import build_app
from backend.services.alerts import AlertCenter
from backend.services.hub import CameraHub
from backend.services.state import DashboardState
from computer_vision.core.config import load_config


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def event(kind="zone_entered", severity="warning", track=1, zone="lab", module="restricted_zone"):
    return {
        "event": kind,
        "module": module,
        "frame_id": 1,
        "timestamp": "2026-10-10T10:00:00+00:00",
        "severity": severity,
        "subject": {"track_id": track, "class": "person"},
        "data": {"zone": zone},
    }


class TestAlertCenter:
    def test_below_min_severity_is_not_an_alert(self):
        center = AlertCenter(min_severity="warning")
        assert center.ingest("cam", [event(severity="info")]) == []
        assert center.ingest("cam", [event(severity="critical")]) == [1]
        assert center.listing()["open_critical"] == 1

    def test_repeats_fold_into_the_open_alert(self):
        clock = Clock()
        center = AlertCenter(dedupe_seconds=30, clock=clock)
        center.ingest("cam", [event()])
        clock.t = 20
        center.ingest("cam", [event(severity="critical")])
        listing = center.listing()
        assert listing["total"] == 1
        alert = listing["alerts"][0]
        assert alert["count"] == 2 and alert["severity"] == "critical"

    def test_different_subject_camera_or_late_repeat_is_new(self):
        clock = Clock()
        center = AlertCenter(dedupe_seconds=30, clock=clock)
        center.ingest("cam", [event(track=1)])
        center.ingest("cam", [event(track=2)])
        center.ingest("other", [event(track=1)])
        clock.t = 31
        center.ingest("cam", [event(track=1)])
        assert center.listing()["total"] == 4

    def test_acknowledged_alert_does_not_absorb_repeats(self):
        center = AlertCenter()
        (first,) = center.ingest("cam", [event()])
        assert center.acknowledge(first)["status"] == "acknowledged"
        (second,) = center.ingest("cam", [event()])
        assert second != first
        assert center.listing(status="open")["alerts"][0]["id"] == second

    def test_acknowledge_all_by_camera(self):
        center = AlertCenter()
        center.ingest("a", [event(track=1), event(track=2)])
        center.ingest("b", [event(track=1)])
        assert center.acknowledge_all("a") == 2
        assert center.listing()["open_count"] == 1
        assert center.acknowledge_all() == 1
        assert center.acknowledge(999) is None

    def test_bounded_and_drops_acknowledged_first(self):
        center = AlertCenter(max_alerts=2, dedupe_seconds=0)
        (a,) = center.ingest("cam", [event(track=1)])
        (b,) = center.ingest("cam", [event(track=2)])
        center.acknowledge(b)
        center.ingest("cam", [event(track=3)])
        ids = {x["id"] for x in center.listing()["alerts"]}
        assert ids == {a, 3}

    @pytest.mark.parametrize(
        "kwargs", [{"min_severity": "loud"}, {"dedupe_seconds": -1}, {"max_alerts": 0}]
    )
    def test_bad_settings_rejected(self, kwargs):
        with pytest.raises(ValueError):
            AlertCenter(**kwargs)

    def test_config_section_loads(self):
        config = load_config("configs/config.yaml", use_env=False)
        center = AlertCenter.from_config(config.dashboard.alerts)
        assert center.min_severity == "warning"


class TestWiring:
    def test_published_events_reach_the_hub_alert_center(self):
        hub = CameraHub()
        hub.configure([("gate", "Gate", "webcam")])
        hub.get("gate").publish(None, {}, {}, [event()])
        listing = hub.alerts.listing()
        assert listing["open_count"] == 1
        assert listing["alerts"][0]["camera"] == "gate"

    def test_broken_alert_handler_does_not_break_publishing(self):
        def boom(events):
            raise RuntimeError("nope")

        state = DashboardState(on_events=boom)
        state.publish(None, {}, {}, [event()])
        assert state.last_event_id == 1


class TestApi:
    def client(self):
        state = DashboardState()
        app = build_app(state)
        return state, TestClient(app, base_url="http://127.0.0.1")

    def test_list_and_acknowledge(self):
        state, client = self.client()
        state.publish(None, {}, {}, [event(), event(severity="info", track=9)])
        body = client.get("/api/alerts").json()
        assert body["open_count"] == 1 and len(body["alerts"]) == 1
        alert_id = body["alerts"][0]["id"]

        acked = client.post(f"/api/alerts/{alert_id}/ack")
        assert acked.status_code == 200 and acked.json()["status"] == "acknowledged"
        assert client.get("/api/alerts?status=open").json()["alerts"] == []
        assert client.post("/api/alerts/12345/ack").status_code == 404

    def test_acknowledge_all(self):
        state, client = self.client()
        state.publish(None, {}, {}, [event(track=1), event(track=2)])
        assert client.post("/api/alerts/ack").json() == {"acknowledged": 2}

    def test_bad_status_is_rejected(self):
        _, client = self.client()
        assert client.get("/api/alerts?status=bogus").status_code == 422
