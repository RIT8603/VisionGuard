"""Dashboard: shared state, the web sink, and the HTTP API.

The API is exercised against a real FastAPI app via TestClient, but never a
real camera or model.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from computer_vision.core.config import ConfigError, build_config
from computer_vision.core.context import BBox, Detection, Event, FrameContext, Severity
from backend.services.web_sink import WebSink
from computer_vision.core.output import MultiSink, NullSink
from backend.services.state import DashboardState
from computer_vision.utils.timing import PipelineStats

fastapi = pytest.importorskip("fastapi", reason="dashboard extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.server import build_app  # noqa: E402


def frame_ctx(frame_id: int = 1, events=()) -> FrameContext:
    ctx = FrameContext(
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=0.0,
    )
    ctx.add_detections(
        "person_detection", [Detection("person", 0.9, BBox(10, 10, 60, 100))]
    )
    for event in events:
        ctx.add_event(event)
    return ctx


def an_event(event_type: str = "zone_entry", severity=Severity.WARNING) -> Event:
    return Event(
        event_type=event_type,
        source_module="restricted_zone",
        frame_id=1,
        timestamp="2026-09-24T10:00:00+00:00",
        severity=severity,
        subject={"track_id": 1, "class": "person"},
        data={"zone": "bay"},
    )


@pytest.fixture
def state() -> DashboardState:
    st = DashboardState(max_events=10)
    st.info = {"source_type": "webcam", "modules": []}
    return st


@pytest.fixture
def client(state) -> TestClient:
    return TestClient(build_app(state))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class TestDashboardConfig:
    def test_disabled_by_default(self):
        assert build_config({}).dashboard.enabled is False

    def test_binds_to_localhost_by_default(self):
        """It has no authentication, so it must not default to the network."""
        assert build_config({}).dashboard.host == "127.0.0.1"

    @pytest.mark.parametrize("port", [0, 70000, -1])
    def test_invalid_port_is_rejected(self, port):
        with pytest.raises(ConfigError, match="port"):
            build_config({"dashboard": {"port": port}})

    @pytest.mark.parametrize("quality", [0, 101])
    def test_invalid_quality_is_rejected(self, quality):
        with pytest.raises(ConfigError, match="jpeg_quality"):
            build_config({"dashboard": {"jpeg_quality": quality}})

    def test_negative_stream_fps_is_rejected(self):
        with pytest.raises(ConfigError, match="stream_fps"):
            build_config({"dashboard": {"stream_fps": -1}})

    def test_env_can_enable_it(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD", "true")
        monkeypatch.setenv("DASHBOARD_PORT", "9999")
        config = build_config({})
        from computer_vision.core.config import apply_env_overrides, build_config as bc

        config = bc(apply_env_overrides({}))
        assert config.dashboard.enabled is True
        assert config.dashboard.port == 9999


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class TestDashboardState:
    def test_starts_empty(self, state):
        assert state.latest_jpeg() is None
        assert state.events_since() == []
        assert state.snapshot()["event_count"] == 0

    def test_publish_replaces_the_frame(self, state):
        state.publish(b"one", {"frame_id": 1}, {"fps": 10}, [])
        state.publish(b"two", {"frame_id": 2}, {"fps": 11}, [])

        assert state.latest_jpeg() == b"two"
        assert state.snapshot()["frame"]["frame_id"] == 2

    def test_only_the_latest_frame_is_kept(self, state):
        """A queue would drift further behind the longer it ran."""
        for i in range(100):
            state.publish(f"f{i}".encode(), {}, {}, [])
        assert state.latest_jpeg() == b"f99"

    def test_events_accumulate_with_ids(self, state):
        state.publish(None, {}, {}, [{"event": "a"}, {"event": "b"}])

        events = state.events_since()
        assert [e["event"] for e in events] == ["b", "a"]  # newest first
        assert [e["id"] for e in events] == [2, 1]

    def test_events_since_filters(self, state):
        state.publish(None, {}, {}, [{"event": "a"}, {"event": "b"}])
        state.publish(None, {}, {}, [{"event": "c"}])

        assert [e["event"] for e in state.events_since(since=2)] == ["c"]
        assert state.events_since(since=3) == []

    def test_event_buffer_is_bounded(self, state):
        for i in range(50):
            state.publish(None, {}, {}, [{"event": f"e{i}"}])

        assert len(state.events_since(limit=200)) == 10  # max_events
        assert state.snapshot()["event_count"] == 50     # total still counted

    def test_publishing_without_a_frame_keeps_the_old_one(self, state):
        state.publish(b"img", {}, {}, [])
        state.publish(None, {"frame_id": 7}, {}, [])

        assert state.latest_jpeg() == b"img"
        assert state.snapshot()["frame"]["frame_id"] == 7

    def test_snapshot_is_a_copy(self, state):
        state.publish(None, {"frame_id": 1}, {"fps": 5}, [])
        snap = state.snapshot()
        snap["stats"]["fps"] = 999

        assert state.snapshot()["stats"]["fps"] == 5


class TestFrameWaiting:
    def test_returns_a_newer_frame(self, state):
        state.publish(b"a", {}, {}, [])
        jpeg, seq = state.wait_for_frame(0, timeout=0.1)
        assert jpeg == b"a" and seq == 1

    def test_times_out_without_a_new_frame(self, state):
        state.publish(b"a", {}, {}, [])
        started = time.perf_counter()
        jpeg, seq = state.wait_for_frame(1, timeout=0.15)

        assert jpeg is None and seq == 1
        assert time.perf_counter() - started >= 0.1

    def test_wakes_when_a_frame_arrives(self, state):
        result = {}

        def waiter():
            result["value"] = state.wait_for_frame(0, timeout=2.0)

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        state.publish(b"late", {}, {}, [])
        thread.join(timeout=2.0)

        assert result["value"][0] == b"late"

    def test_stop_releases_waiters(self, state):
        """Otherwise shutdown hangs on an open stream."""
        result = {}

        def waiter():
            result["value"] = state.wait_for_frame(0, timeout=5.0)

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        state.stop()
        thread.join(timeout=2.0)

        assert not thread.is_alive()


# --------------------------------------------------------------------------- #
# The sink
# --------------------------------------------------------------------------- #

class TestWebSink:
    def _sink(self, state, **kwargs):
        config = build_config({"processing": {"show_window": False}}).processing
        return WebSink(state, config, **kwargs)

    def test_publishes_an_encoded_frame(self, state):
        sink = self._sink(state, stream_fps=0)
        sink.emit(frame_ctx(), PipelineStats())

        jpeg = state.latest_jpeg()
        assert jpeg is not None
        assert jpeg[:2] == b"\xff\xd8"  # JPEG magic

    def test_publishes_results_and_events(self, state):
        sink = self._sink(state, stream_fps=0)
        sink.emit(frame_ctx(events=[an_event()]), PipelineStats())

        snap = state.snapshot()
        assert snap["frame"]["detections"]["person_detection"][0]["class"] == "person"
        assert state.events_since()[0]["event"] == "zone_entry"

    def test_stream_fps_throttles_encoding_only(self, state):
        """Analysis must keep full rate even when the view is capped."""
        sink = self._sink(state, stream_fps=1.0)

        sink.emit(frame_ctx(1, events=[an_event()]), PipelineStats())
        first = state.latest_jpeg()
        sink.emit(frame_ctx(2, events=[an_event("zone_exit")]), PipelineStats())

        # Second frame was too soon to re-encode...
        assert state.latest_jpeg() is first
        # ...but its results and events were still published.
        assert state.snapshot()["frame"]["frame_id"] == 2
        assert len(state.events_since()) == 2

    def test_never_asks_the_pipeline_to_stop(self, state):
        """A browser must not be able to end a run."""
        assert self._sink(state).emit(frame_ctx(), PipelineStats()) is True

    def test_close_stops_the_state(self, state):
        sink = self._sink(state)
        sink.close()
        assert state.running is False

    def test_jpeg_quality_changes_size(self, state):
        # A noisy frame so quality actually matters.
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)

        sizes = []
        for quality in (20, 95):
            st = DashboardState()
            sink = self._sink(st, stream_fps=0, jpeg_quality=quality)
            ctx = frame_ctx()
            ctx.frame = noisy
            sink.emit(ctx, PipelineStats())
            sizes.append(len(st.latest_jpeg()))

        assert sizes[0] < sizes[1]


class TestMultiSink:
    def test_fans_out_to_every_sink(self, state):
        config = build_config({"processing": {"show_window": False}}).processing
        web = WebSink(state, config, stream_fps=0)
        multi = MultiSink([NullSink(log_every=0), web])

        assert multi.emit(frame_ctx(), PipelineStats()) is True
        assert state.latest_jpeg() is not None

    def test_any_sink_can_stop_the_run(self):
        class Stopper(NullSink):
            def emit(self, context, stats):
                return False

        multi = MultiSink([NullSink(log_every=0), Stopper(log_every=0)])
        assert multi.emit(frame_ctx(), PipelineStats()) is False

    def test_all_sinks_still_receive_the_frame_when_one_stops(self, state):
        class Stopper(NullSink):
            def emit(self, context, stats):
                return False

        config = build_config({"processing": {"show_window": False}}).processing
        web = WebSink(state, config, stream_fps=0)
        MultiSink([Stopper(log_every=0), web]).emit(frame_ctx(), PipelineStats())

        assert state.latest_jpeg() is not None

    def test_a_failing_close_does_not_block_the_others(self, state):
        class Bad(NullSink):
            def close(self):
                raise RuntimeError("nope")

        config = build_config({"processing": {"show_window": False}}).processing
        web = WebSink(state, config)
        MultiSink([Bad(log_every=0), web]).close()

        assert state.running is False


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #

class TestApi:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "VisionGuard" in response.text
        assert "/stream.mjpg" in response.text

    def test_status_reports_the_pipeline(self, client, state):
        state.publish(None, {"frame_id": 12}, {"fps": 24.5}, [])

        body = client.get("/api/status").json()

        assert body["running"] is True
        assert body["stats"]["fps"] == 24.5
        assert body["frame"]["frame_id"] == 12
        assert body["info"]["source_type"] == "webcam"

    def test_events_endpoint(self, client, state):
        state.publish(None, {}, {}, [an_event().to_dict()])

        events = client.get("/api/events").json()["events"]

        assert len(events) == 1
        assert events[0]["event"] == "zone_entry"
        assert events[0]["severity"] == "warning"

    def test_events_since_filters(self, client, state):
        state.publish(None, {}, {}, [an_event().to_dict()])
        state.publish(None, {}, {}, [an_event("zone_exit").to_dict()])

        events = client.get("/api/events?since=1").json()["events"]

        assert [e["event"] for e in events] == ["zone_exit"]

    def test_events_limit_is_bounded(self, client):
        assert client.get("/api/events?limit=9999").status_code == 422
        assert client.get("/api/events?since=-1").status_code == 422

    def test_snapshot_before_any_frame(self, client):
        assert client.get("/api/snapshot.jpg").status_code == 503

    def test_snapshot_returns_jpeg(self, client, state):
        state.publish(b"\xff\xd8fake", {}, {}, [])

        response = client.get("/api/snapshot.jpg")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"\xff\xd8fake"

    def test_stream_is_multipart(self, client, state):
        state.publish(b"\xff\xd8frame", {}, {}, [])
        state.stop()  # so the generator terminates for the test

        with client.stream("GET", "/stream.mjpg") as response:
            assert response.status_code == 200
            assert "multipart/x-mixed-replace" in response.headers["content-type"]

    def test_api_is_read_only(self, client):
        """No endpoint may reconfigure or control the pipeline."""
        for method, path in (
            ("post", "/api/status"),
            ("delete", "/api/events"),
            ("post", "/"),
        ):
            assert getattr(client, method)(path).status_code == 405
