"""Multi-camera: config, storage, API, thread safety, and a real two-camera run."""

from __future__ import annotations

import socket
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from backend.api.server import build_app
from backend.main import initial_cameras, main
from backend.services.camera_manager import CameraManager
from backend.services.cameras import (
    MASK,
    cameras_file_for,
    load_camera_file,
    mask_url,
    parse_cameras,
    public_view,
    save_camera_file,
)
from backend.services.hub import PHASE_ENDED, PHASE_FAILED, CameraHub
from backend.services.reload import ReloadController
from backend.services.runtime import (
    build_runtimes,
    describe_specs,
    run_group,
    specs_from_cameras,
    specs_from_video,
)
from computer_vision.core.config import (
    CameraConfig,
    ConfigError,
    VideoConfig,
    build_config,
    load_config,
)
from computer_vision.core.pipeline import RunLimits
from computer_vision.inference.base import DetectorBackend, RawDetection
from computer_vision.inference.shared import SharedDetector
def write_clip(path: Path, frames: int = 15) -> Path:
    """A short clip; brightness wraps so any frame count is valid."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("No MJPG encoder available in this OpenCV build")
    for index in range(frames):
        writer.write(np.full((120, 160, 3), (index * 7) % 250, dtype=np.uint8))
    writer.release()
    return path

SECRET = "hunter2-very-secret"
RTSP = f"rtsp://admin:{SECRET}@10.0.0.5:554/stream1"


def cam(id_: str = "gate", **kwargs) -> CameraConfig:
    kwargs.setdefault("source", "rtsp")
    if kwargs["source"] == "rtsp":
        kwargs.setdefault("rtsp_url", RTSP)
    return CameraConfig(id=id_, **kwargs)


# --------------------------------------------------------------------------- #
# CameraConfig
# --------------------------------------------------------------------------- #


class TestCameraConfig:
    def test_id_is_normalised_and_used_as_the_source_id(self):
        camera = cam("Gate-1")
        assert camera.id == "gate-1"
        assert camera.to_video(VideoConfig()).resolved_source_id() == "gate-1"

    @pytest.mark.parametrize("bad", ["", "a b", "../x", "-x", "x" * 33, "ünï"])
    def test_bad_ids_are_rejected(self, bad):
        with pytest.raises(ValueError):
            cam(bad)

    @pytest.mark.parametrize(
        "url", ["http://h/x", "file:///etc/passwd", "concat:a|b", "rtsp:/nohost", "ftp://h"]
    )
    def test_only_rtsp_urls_are_accepted(self, url):
        with pytest.raises(ValueError, match="rtsp"):
            cam(rtsp_url=url)

    def test_rtsps_is_accepted(self):
        assert cam(rtsp_url="rtsps://h/x").rtsp_url == "rtsps://h/x"

    def test_source_specific_fields_are_required(self):
        with pytest.raises(ValueError):
            CameraConfig(id="a", source="rtsp")
        with pytest.raises(ValueError):
            CameraConfig(id="a", source="file")
        assert CameraConfig(id="a", source="webcam", device_id=2).device_id == 2

    def test_shared_video_settings_carry_over(self):
        base = VideoConfig(rtsp_transport="udp", width=640, height=360, reconnect_attempts=9)
        video = cam().to_video(base)
        assert (video.rtsp_transport, video.width, video.reconnect_attempts) == ("udp", 640, 9)
        assert video.rtsp_url == RTSP

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValueError):
            CameraConfig(id="a", source="webcam", surprise=1)


# --------------------------------------------------------------------------- #
# Masking and payload parsing
# --------------------------------------------------------------------------- #


class TestMasking:
    def test_password_and_secret_query_values_are_masked(self):
        masked = mask_url("rtsp://u:pw@h:554/p?token=abc&channel=1")
        assert "pw" not in masked and "abc" not in masked
        assert masked == f"rtsp://u:{MASK}@h:554/p?token={MASK}&channel=1"

    def test_url_without_secrets_is_unchanged(self):
        assert mask_url("rtsp://10.0.0.5/stream") == "rtsp://10.0.0.5/stream"

    def test_public_view_never_contains_the_password(self):
        assert SECRET not in str(public_view(cam()))


class TestParseCameras:
    def test_ids_are_generated_and_kept_unique(self):
        cams, errors = parse_cameras(
            [
                {"source": "webcam", "device_id": 0},
                {"id": "cam-1", "source": "webcam", "device_id": 1},
                {"source": "rtsp", "rtsp_url": "rtsp://h/x"},
            ],
            max_cameras=8,
        )
        assert errors == []
        assert len({c.id for c in cams}) == 3

    def test_empty_and_too_many_are_rejected(self):
        assert parse_cameras([], max_cameras=8)[1]
        too_many = [{"source": "webcam", "device_id": i} for i in range(3)]
        assert "at most 2" in parse_cameras(too_many, max_cameras=2)[1][0]

    def test_all_problems_are_reported_together(self):
        _, errors = parse_cameras(
            [{"source": "rtsp"}, {"source": "file"}, "nonsense"], max_cameras=8
        )
        assert len(errors) == 3
        assert errors[0].startswith("camera 1")

    def test_duplicate_ids_and_webcam_devices_are_rejected(self):
        _, errors = parse_cameras(
            [
                {"id": "a", "source": "webcam", "device_id": 0},
                {"id": "a", "source": "webcam", "device_id": 1},
            ],
            max_cameras=8,
        )
        assert "duplicate id" in errors[0]
        _, errors = parse_cameras(
            [
                {"id": "a", "source": "webcam", "device_id": 0},
                {"id": "b", "source": "webcam", "device_id": 0},
            ],
            max_cameras=8,
        )
        assert "used twice" in errors[0]

    def test_a_masked_password_is_restored_from_the_running_camera(self):
        existing = [cam("gate")]
        cams, errors = parse_cameras(
            [{"id": "gate", "source": "rtsp", "rtsp_url": mask_url(RTSP)}],
            max_cameras=8,
            existing=existing,
        )
        assert errors == [] and cams[0].rtsp_url == RTSP

    def test_a_masked_password_survives_an_edit_to_the_rest_of_the_url(self):
        edited = mask_url(RTSP).replace("stream1", "stream2")
        cams, errors = parse_cameras(
            [{"id": "gate", "source": "rtsp", "rtsp_url": edited}],
            max_cameras=8,
            existing=[cam("gate")],
        )
        assert errors == [] and cams[0].rtsp_url == RTSP.replace("stream1", "stream2")

    def test_a_masked_password_with_nothing_to_restore_is_an_error(self):
        _, errors = parse_cameras(
            [{"id": "new", "source": "rtsp", "rtsp_url": mask_url(RTSP)}], max_cameras=8
        )
        assert errors and "password is hidden" in errors[0]

    def test_error_messages_never_echo_a_secret(self):
        _, errors = parse_cameras(
            [{"id": "x", "source": "rtsp", "rtsp_url": f"http://u:{SECRET}@h/x"}],
            max_cameras=8,
        )
        assert errors and SECRET not in " ".join(errors)


# --------------------------------------------------------------------------- #
# cameras.yaml
# --------------------------------------------------------------------------- #


class TestCameraFile:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "cameras.yaml"
        cams = [
            cam("gate", name="Main gate"),
            CameraConfig(id="lab", source="webcam", device_id=1),
            CameraConfig(id="clip", source="file", path="a.mp4", loop=True),
        ]
        save_camera_file(path, cams)
        assert load_camera_file(path) == cams

    def test_file_is_owner_only_where_supported(self, tmp_path):
        path = tmp_path / "cameras.yaml"
        save_camera_file(path, [cam()])
        import os
        import sys

        if sys.platform != "win32":
            assert os.stat(path).st_mode & 0o077 == 0

    def test_no_stray_temp_files_are_left(self, tmp_path):
        save_camera_file(tmp_path / "cameras.yaml", [cam()])
        assert [p.name for p in tmp_path.iterdir()] == ["cameras.yaml"]

    def test_missing_file_is_none(self, tmp_path):
        assert load_camera_file(tmp_path / "nope.yaml") is None

    @pytest.mark.parametrize("text", ["cameras: []", "cameras: nope", "{not yaml", "[]"])
    def test_a_broken_file_is_an_error_not_a_silent_fallback(self, tmp_path, text):
        path = tmp_path / "cameras.yaml"
        path.write_text(text)
        with pytest.raises(ConfigError):
            load_camera_file(path)

    def test_file_error_does_not_leak_secrets(self, tmp_path):
        path = tmp_path / "cameras.yaml"
        path.write_text(f"cameras:\n  - id: a\n    source: rtsp\n    rtsp_url: http://u:{SECRET}@h/\n")
        with pytest.raises(ConfigError) as caught:
            load_camera_file(path)
        assert SECRET not in str(caught.value)


class TestInitialCameras:
    def _args(self, *argv):
        from backend.main import build_parser

        return build_parser().parse_args(list(argv))

    def _config(self, tmp_path: Path):
        path = tmp_path / "config.yaml"
        path.write_text("video:\n  source: webcam\n")
        return path, load_config(str(path))

    def test_no_file_means_the_single_video_camera(self, tmp_path):
        path, config = self._config(tmp_path)
        specs, origin = initial_cameras(self._args("--config", str(path)), config, path)
        assert origin == "video" and [s.id for s in specs] == ["webcam-0"]

    def test_cameras_file_is_used_when_present(self, tmp_path):
        path, config = self._config(tmp_path)
        save_camera_file(cameras_file_for(path), [cam("a"), cam("b")])
        specs, origin = initial_cameras(self._args(), config, path)
        assert origin == "file" and [s.id for s in specs] == ["a", "b"]

    def test_a_camera_on_the_command_line_wins_over_the_file(self, tmp_path):
        path, config = self._config(tmp_path)
        save_camera_file(cameras_file_for(path), [cam("a"), cam("b")])
        for flag in (["--device-id", "1"], ["--rtsp-url", "rtsp://h/x"], ["--source", "webcam"]):
            specs, origin = initial_cameras(self._args(*flag), config, path)
            assert origin == "cli" and len(specs) == 1

    def test_a_broken_file_stops_startup(self, tmp_path):
        path, config = self._config(tmp_path)
        cameras_file_for(path).write_text("cameras: []")
        with pytest.raises(ConfigError):
            initial_cameras(self._args(), config, path)


# --------------------------------------------------------------------------- #
# ReloadController (camera requests)
# --------------------------------------------------------------------------- #


class TestReloadCameras:
    def test_camera_request_round_trip(self):
        reload = ReloadController()
        assert reload.pop_cameras() is None
        reload.request_cameras([cam("a")])
        assert reload.pending
        assert [c.id for c in reload.pop_cameras()] == ["a"]
        assert not reload.pending

    def test_the_wake_up_stays_until_both_kinds_are_consumed(self):
        reload = ReloadController()
        reload.request(["person_detection"])
        reload.request_cameras([cam("a")])
        reload.pop()
        assert reload.pending  # a camera change is still waiting
        reload.pop_cameras()
        assert not reload.pending

    def test_wait_for_times_out_when_idle(self):
        assert ReloadController().wait_for(0.01) is False


# --------------------------------------------------------------------------- #
# Hub
# --------------------------------------------------------------------------- #


class TestHub:
    def test_states_survive_reconfiguration_by_id(self):
        hub = CameraHub()
        hub.configure([("a", "A", "rtsp"), ("b", "B", "rtsp")])
        state_a = hub.get("a")
        hub.configure([("b", "B", "rtsp"), ("a", "A2", "rtsp")])
        assert hub.get("a") is state_a
        assert hub.default_id == "b"
        assert hub.ids() == ["b", "a"]

    def test_removed_cameras_are_gone(self):
        hub = CameraHub()
        hub.configure([("a", "A", "rtsp"), ("b", "B", "rtsp")])
        hub.configure([("b", "B", "rtsp")])
        assert hub.get("a") is None

    def test_first_camera_is_the_default(self):
        hub = CameraHub()
        hub.configure([("a", "A", "rtsp"), ("b", "B", "rtsp")])
        assert hub.get(None) is hub.get("a")

    def test_status_follows_the_pipeline(self):
        hub = CameraHub()
        hub.configure([("a", "A", "rtsp")])
        assert hub.describe()[0]["status"] == "starting"

        hub.get("a").publish(b"jpg", {}, {"fps": 7.5}, [])
        row = hub.describe()[0]
        assert row["status"] == "live" and row["fps"] == 7.5

        hub.set_phase("a", PHASE_ENDED)
        assert hub.describe()[0]["status"] == "offline"
        hub.set_phase("a", PHASE_FAILED, "unreachable")
        row = hub.describe()[0]
        assert (row["status"], row["error"]) == ("failed", "unreachable")

    def test_a_silent_camera_is_reported_stalled(self, monkeypatch):
        hub = CameraHub()
        hub.configure([("a", "A", "rtsp")])
        hub.get("a").publish(b"jpg", {}, {}, [])
        monkeypatch.setattr(hub.get("a"), "seconds_since_publish", lambda: 60.0)
        assert hub.describe()[0]["status"] == "stalled"


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@pytest.fixture
def api():
    hub = CameraHub()
    hub.configure([("gate", "Gate", "rtsp"), ("lab", "Lab", "webcam")])
    hub.get("gate").publish(b"\xff\xd8gate", {"frame_id": 1}, {"fps": 5.0}, [{"type": "loitering"}])
    hub.get("lab").publish(b"\xff\xd8lab", {"frame_id": 9}, {"fps": 8.0}, [])
    reload = ReloadController()
    manager = CameraManager(hub, reload, max_cameras=4, editable=True)
    manager.set_current([cam("gate"), CameraConfig(id="lab", source="webcam")])
    app = build_app(hub.get(None), hub=hub, cameras=manager, reload=reload)
    return TestClient(app, base_url="http://127.0.0.1"), reload, manager, hub


class TestCamerasApi:
    def test_listing_has_status_and_masked_urls(self, api):
        client, *_ = api
        body = client.get("/api/cameras").json()
        assert [c["id"] for c in body["cameras"]] == ["gate", "lab"]
        assert body["max_cameras"] == 4 and body["editable"] is True
        gate = body["cameras"][0]
        assert gate["status"] == "live" and gate["rtsp_url"] == mask_url(RTSP)
        assert SECRET not in client.get("/api/cameras").text

    def test_read_endpoints_default_to_the_first_camera(self, api):
        client, *_ = api
        assert client.get("/api/status").json()["frame"]["frame_id"] == 1
        assert client.get("/api/snapshot.jpg").content.endswith(b"gate")

    def test_read_endpoints_can_pick_a_camera(self, api):
        client, *_ = api
        assert client.get("/api/status?camera=lab").json()["frame"]["frame_id"] == 9
        assert client.get("/api/snapshot.jpg?camera=lab").content.endswith(b"lab")
        events = client.get("/api/events?camera=gate").json()["events"]
        assert [e["type"] for e in events] == ["loitering"]
        assert client.get("/api/events?camera=lab").json()["events"] == []

    @pytest.mark.parametrize(
        "path", ["/api/status", "/api/events", "/api/snapshot.jpg", "/stream.mjpg"]
    )
    def test_unknown_camera_is_404(self, api, path):
        client, *_ = api
        assert client.get(f"{path}?camera=nope").status_code == 404

    def test_post_queues_a_valid_list(self, api):
        client, reload, *_ = api
        response = client.post(
            "/api/cameras",
            json={"cameras": [{"id": "a", "source": "rtsp", "rtsp_url": RTSP}, {"source": "webcam"}]},
        )
        assert response.status_code == 202
        assert SECRET not in response.text
        queued = reload.pop_cameras()
        assert [c.id for c in queued] == ["a", "cam-1"]
        assert queued[0].rtsp_url == RTSP

    def test_post_restores_a_masked_password(self, api):
        client, reload, *_ = api
        client.post(
            "/api/cameras",
            json={"cameras": [{"id": "gate", "source": "rtsp", "rtsp_url": mask_url(RTSP)}]},
        )
        assert reload.pop_cameras()[0].rtsp_url == RTSP

    def test_post_rejects_bad_input_without_echoing_secrets(self, api):
        client, reload, *_ = api
        response = client.post(
            "/api/cameras",
            json={"cameras": [{"id": "x", "source": "rtsp", "rtsp_url": f"http://u:{SECRET}@h/"}]},
        )
        assert response.status_code == 400
        assert SECRET not in response.text
        assert reload.pop_cameras() is None

    @pytest.mark.parametrize("body", [{}, {"cameras": "x"}, {"cameras": []}, {"cameras": [1]}])
    def test_post_rejects_malformed_bodies(self, api, body):
        client, reload, *_ = api
        assert client.post("/api/cameras", json=body).status_code == 400
        assert reload.pop_cameras() is None

    def test_post_rejects_more_than_the_maximum(self, api):
        client, reload, *_ = api
        many = [{"source": "webcam", "device_id": i} for i in range(5)]
        response = client.post("/api/cameras", json={"cameras": many})
        assert response.status_code == 400 and "at most 4" in response.json()["error"]

    def test_editing_can_be_turned_off(self):
        hub = CameraHub()
        hub.configure([("a", "A", "webcam")])
        manager = CameraManager(hub, ReloadController(), max_cameras=4, editable=False)
        client = TestClient(build_app(hub.get(None), hub=hub, cameras=manager), base_url="http://127.0.0.1")
        assert client.get("/api/cameras").json()["editable"] is False
        response = client.post("/api/cameras", json={"cameras": [{"source": "webcam"}]})
        assert response.status_code == 403

    def test_without_a_manager_post_is_503(self):
        client = TestClient(build_app(CameraHub().get(None) or __import__("backend.services.state", fromlist=["DashboardState"]).DashboardState()), base_url="http://127.0.0.1")
        assert client.post("/api/cameras", json={"cameras": []}).status_code == 503

    def test_apply_error_is_exposed(self, api):
        client, reload, *_ = api
        reload.fail("model missing")
        assert client.get("/api/cameras").json()["apply_error"] == "model missing"

    def test_host_check_still_applies(self, api):
        client, *_ = api
        response = client.post(
            "/api/cameras",
            json={"cameras": [{"source": "webcam"}]},
            headers={"Host": "evil.example"},
        )
        assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Shared detector across threads
# --------------------------------------------------------------------------- #


class _Probe(DetectorBackend):
    """Counts calls and the most that ever ran at once."""

    name = "probe"

    def __init__(self):
        super().__init__({})
        self.calls = 0
        self.running = 0
        self.max_running = 0
        self._guard = threading.Lock()

    def load(self):
        self._loaded = True

    def infer(self, frame):
        with self._guard:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(0.01)
        with self._guard:
            self.running -= 1
            self.calls += 1
        return [RawDetection(label="person", confidence=0.9, xyxy=(0, 0, 1, 1), class_id=0)]


class TestSharedDetectorThreads:
    def test_inference_is_never_concurrent(self):
        probe = _Probe()
        shared = SharedDetector(probe)
        shared.load()

        def camera():
            for _ in range(10):
                shared.infer(np.zeros((8, 8, 3), np.uint8))

        threads = [threading.Thread(target=camera) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert probe.max_running == 1 and probe.calls == 40

    def test_another_camera_cannot_evict_this_cameras_memo(self):
        probe = _Probe()
        shared = SharedDetector(probe)
        shared.load()
        frame_a = np.zeros((8, 8, 3), np.uint8)
        frame_b = np.ones((8, 8, 3), np.uint8)

        shared.infer(frame_a)  # module 1 of camera A
        other = threading.Thread(target=lambda: shared.infer(frame_b))  # camera B in between
        other.start()
        other.join()
        shared.infer(frame_a)  # module 2 of camera A: must hit A's memo

        assert probe.calls == 2
        assert shared.stats()["cache_hits"] == 1


# --------------------------------------------------------------------------- #
# Real runs
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def clips(tmp_path):
    return write_clip(tmp_path / "one.avi", frames=15), write_clip(tmp_path / "two.avi", frames=15)


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            processing:
              show_window: false
              target_width: null
              stall_timeout_sec: 5
            modules:
              person_detection:
                enabled: true
                backend: fake
            {extra}
            """
        )
    )
    return path


class TestRunGroup:
    def _specs(self, config, clips, **overrides):
        cams = [
            CameraConfig(id="one", source="file", path=str(clips[0])),
            CameraConfig(id="two", source="file", path=str(clips[1])),
        ]
        return specs_from_cameras(config, cams)

    def test_two_cameras_run_independently_with_their_own_modules(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        specs = self._specs(config, clips)
        runtimes = build_runtimes(config, specs)

        assert runtimes[0].modules[0] is not runtimes[1].modules[0]
        # ...but they share one loaded model.
        assert runtimes[0].modules[0]._backend is runtimes[1].modules[0]._backend

        assert run_group(runtimes, RunLimits()) is False
        assert [rt.stats.frames_processed for rt in runtimes] == [15, 15]
        assert all(rt.failure is None for rt in runtimes)

    def test_frames_are_stamped_with_their_camera_id(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        runtimes = build_runtimes(config, self._specs(config, clips))
        seen: dict[str, set[str]] = {"one": set(), "two": set()}
        for rt in runtimes:
            original = rt.sink.emit

            def spy(context, stats, _id=rt.id, _orig=original):
                seen[_id].add(context.source_id)
                return _orig(context, stats)

            rt.sink.emit = spy
        run_group(runtimes, RunLimits())
        assert seen == {"one": {"one"}, "two": {"two"}}

    def test_one_failing_camera_does_not_stop_the_other(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        cams = [
            CameraConfig(id="good", source="file", path=str(clips[0])),
            CameraConfig(id="bad", source="file", path=str(tmp_path / "missing.avi")),
        ]
        hub = CameraHub()
        hub.configure([("good", "good", "file"), ("bad", "bad", "file")])
        runtimes = build_runtimes(config, specs_from_cameras(config, cams), hub)
        run_group(runtimes, RunLimits(), hub=hub)

        good, bad = runtimes
        assert good.stats.frames_processed == 15 and good.failure is None
        assert bad.failure is not None
        rows = {row["id"]: row for row in hub.describe()}
        assert rows["good"]["status"] == "offline"
        assert rows["bad"]["status"] == "failed" and rows["bad"]["error"]

    def test_a_reload_request_stops_every_camera(self, tmp_path):
        looping = [
            CameraConfig(id="a", source="file", path=str(write_clip(tmp_path / "a.avi", 30)), loop=True),
            CameraConfig(id="b", source="file", path=str(write_clip(tmp_path / "b.avi", 30)), loop=True),
        ]
        config = load_config(str(_config(tmp_path)))
        runtimes = build_runtimes(config, specs_from_cameras(config, looping))
        reload = ReloadController()
        threading.Timer(0.5, lambda: reload.request(["person_detection"])).start()

        started = time.monotonic()
        run_group(runtimes, RunLimits(), reload)
        assert time.monotonic() - started < 8
        assert all(rt.stats.frames_processed > 0 for rt in runtimes)

    def test_a_single_camera_runs_on_the_calling_thread(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        spec = specs_from_cameras(config, [CameraConfig(id="solo", source="file", path=str(clips[0]))])
        runtimes = build_runtimes(config, spec)
        names = []
        original = runtimes[0].sink.emit

        def spy(context, stats):
            names.append(threading.current_thread() is threading.main_thread())
            return original(context, stats)

        runtimes[0].sink.emit = spy
        run_group(runtimes, RunLimits())
        assert names and all(names)

    def test_a_failed_build_releases_everything(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        specs = self._specs(config, clips)
        FakeBackend = __import__("tools.tests.conftest", fromlist=["FakeDetectorBackend"]).FakeDetectorBackend
        FakeBackend.fail_on_load = True
        with pytest.raises(Exception):
            build_runtimes(config, specs)
        FakeBackend.fail_on_load = False
        # A later build must work: nothing half-built is left behind.
        assert len(build_runtimes(config, specs)) == 2

    def test_window_is_disabled_for_several_cameras(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        config = config.model_copy(update={"processing": config.processing.model_copy(update={"show_window": True})})
        runtimes = build_runtimes(config, self._specs(config, clips))
        assert all(rt.config.processing.show_window is False for rt in runtimes)

    def test_describe_specs_masks_a_video_camera(self):
        config = build_config({"video": {"source": "rtsp", "rtsp_url": RTSP}})
        cameras, details = describe_specs(specs_from_video(config))
        assert cameras[0].rtsp_url == RTSP and details == {}
        weird = build_config({"video": {"source": "rtsp", "rtsp_url": f"http://u:{SECRET}@h/x"}})
        cameras, details = describe_specs(specs_from_video(weird))
        assert cameras == [] and SECRET not in str(details)


class TestMainEndToEnd:
    def test_two_cameras_from_cameras_yaml(self, tmp_path, clips):
        config = _config(tmp_path)
        save_camera_file(
            cameras_file_for(config),
            [
                CameraConfig(id="one", source="file", path=str(clips[0])),
                CameraConfig(id="two", source="file", path=str(clips[1])),
            ],
        )
        assert main(["--config", str(config), "--no-window"]) == 0

    def test_all_cameras_failing_is_an_error_exit(self, tmp_path):
        config = _config(tmp_path)
        save_camera_file(
            cameras_file_for(config),
            [
                CameraConfig(id="a", source="file", path=str(tmp_path / "x.avi")),
                CameraConfig(id="b", source="file", path=str(tmp_path / "y.avi")),
            ],
        )
        assert main(["--config", str(config), "--no-window"]) == 3

    def test_a_single_failing_camera_keeps_the_old_exit_code(self, tmp_path):
        config = _config(tmp_path)
        assert main(["--config", str(config), "--no-window", "--source", "file",
                     "--video-path", str(tmp_path / "missing.avi")]) == 3

    def test_a_broken_cameras_file_is_a_config_error(self, tmp_path):
        config = _config(tmp_path)
        cameras_file_for(config).write_text("cameras: []")
        assert main(["--config", str(config), "--no-window"]) == 2

    def test_dashboard_adds_and_removes_cameras_live(self, tmp_path, clips):
        import httpx

        looping = [
            write_clip(tmp_path / "l1.avi", 30),
            write_clip(tmp_path / "l2.avi", 30),
            write_clip(tmp_path / "l3.avi", 30),
        ]
        config = _config(tmp_path)
        save_camera_file(
            cameras_file_for(config),
            [CameraConfig(id="one", source="file", path=str(looping[0]), loop=True)],
        )
        port = _free_port()
        exit_codes: list[int] = []
        worker = threading.Thread(
            target=lambda: exit_codes.append(
                main(["--config", str(config), "--no-window", "--dashboard",
                      "--dashboard-port", str(port), "--duration", "25"])
            ),
            daemon=True,
        )
        worker.start()
        base = f"http://127.0.0.1:{port}"

        def listing():
            try:
                return httpx.get(f"{base}/api/cameras", timeout=2).json()
            except httpx.HTTPError:
                return None

        def wait_for(predicate, seconds=15):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                body = listing()
                if body and predicate(body):
                    return body
                time.sleep(0.2)
            raise AssertionError(f"timed out; last listing: {listing()}")

        wait_for(lambda b: [c["id"] for c in b["cameras"]] == ["one"] and b["cameras"][0]["status"] == "live")

        response = httpx.post(
            f"{base}/api/cameras",
            json={"cameras": [
                {"id": "one", "source": "file", "path": str(looping[0]), "loop": True},
                {"id": "two", "source": "file", "path": str(looping[1]), "loop": True},
                {"id": "three", "source": "file", "path": str(looping[2]), "loop": True},
            ]},
        )
        assert response.status_code == 202
        body = wait_for(
            lambda b: sorted(c["id"] for c in b["cameras"]) == ["one", "three", "two"]
            and all(c["status"] == "live" for c in b["cameras"])
        )
        assert httpx.get(f"{base}/api/snapshot.jpg?camera=three", timeout=2).status_code == 200
        assert httpx.get(f"{base}/api/status?camera=two", timeout=2).json()["info"]["camera_id"] == "two"

        saved = yaml.safe_load(cameras_file_for(config).read_text())
        assert [c["id"] for c in saved["cameras"]] == ["one", "two", "three"]

        # Back to one camera.
        httpx.post(f"{base}/api/cameras", json={"cameras": [
            {"id": "two", "source": "file", "path": str(looping[1]), "loop": True}]})
        wait_for(lambda b: [c["id"] for c in b["cameras"]] == ["two"] and b["cameras"][0]["status"] == "live")
        assert httpx.get(f"{base}/api/status?camera=one", timeout=2).status_code == 404

        # A request that cannot be built keeps what is running - with fresh,
        # working modules (the old ones were released when their pipelines ended).
        response = httpx.post(f"{base}/api/modules", json={"enabled": ["person_tracking"]})
        assert response.status_code == 202  # accepted; the rebuild fails afterwards
        body = wait_for(lambda b: b["apply_error"] is not None)
        assert "person_detection" in body["apply_error"]
        wait_for(lambda b: b["cameras"][0]["status"] == "live")
        status = httpx.get(f"{base}/api/status?camera=two", timeout=2).json()
        assert [m["name"] for m in status["info"]["modules"]] == ["person_detection"]
        assert "detections" in status["frame"]

        worker.join(timeout=60)
        assert not worker.is_alive()
        assert exit_codes and exit_codes[0] in (0, 3)


# --------------------------------------------------------------------------- #
# Per-camera module selection
# --------------------------------------------------------------------------- #


class TestPerCameraModules:
    def test_default_is_to_follow_the_shared_selection(self):
        assert cam().modules is None and public_view(cam())["modules"] is None

    def test_names_are_cleaned_and_deduplicated(self):
        chosen = cam(modules=[" person_detection ", "person_detection", "low_light"])
        assert chosen.modules == ["person_detection", "low_light"]
        assert public_view(chosen)["modules"] == ["person_detection", "low_light"]

    def test_an_empty_list_is_kept_as_video_only(self):
        assert cam(modules=[]).modules == []

    def test_unknown_names_are_refused_when_a_catalogue_is_given(self):
        item = {"source": "webcam", "modules": ["person_detection", "no_such_module"]}
        _, errors = parse_cameras([item], max_cameras=4, known_modules=["person_detection"])
        assert errors and "no_such_module" in errors[0]
        cams, errors = parse_cameras([item], max_cameras=4)
        assert not errors and cams[0].modules == ["person_detection", "no_such_module"]

    def test_file_round_trip_keeps_none_and_empty_apart(self, tmp_path):
        path = tmp_path / "cameras.yaml"
        cams = [
            cam("a"),
            cam("b", modules=[]),
            cam("c", modules=["person_detection", "person_tracking"]),
        ]
        save_camera_file(path, cams)
        assert load_camera_file(path) == cams
        assert "modules" not in yaml.safe_load(path.read_text())["cameras"][0]

    def test_api_lists_and_accepts_modules(self, api):
        client, reload, manager, _ = api
        manager.set_current(
            [cam("gate", modules=["low_light"]), CameraConfig(id="lab", source="webcam")],
            shared_modules=["person_detection"],
        )
        body = client.get("/api/cameras").json()
        assert body["shared_modules"] == ["person_detection"]
        assert [c["modules"] for c in body["cameras"]] == [["low_light"], None]

        ok = client.post(
            "/api/cameras",
            json={"cameras": [{"id": "lab", "source": "webcam", "modules": ["fall_detection"]}]},
        )
        assert ok.status_code == 202 and ok.json()["cameras"][0]["modules"] == ["fall_detection"]
        assert reload.pop_cameras()[0].modules == ["fall_detection"]

        bad = client.post(
            "/api/cameras",
            json={"cameras": [{"id": "lab", "source": "webcam", "modules": ["nope"]}]},
        )
        assert bad.status_code == 400 and "nope" in bad.json()["error"]

    def test_each_camera_builds_only_its_own_modules(self, tmp_path, clips):
        config = load_config(str(_config(tmp_path)))
        cams = [
            CameraConfig(id="shared", source="file", path=str(clips[0])),
            CameraConfig(id="none", source="file", path=str(clips[1]), modules=[]),
            CameraConfig(
                id="own", source="file", path=str(clips[1]), modules=["person_detection"]
            ),
        ]
        runtimes = build_runtimes(config, specs_from_cameras(config, cams))
        names = [[m.module_name for m in rt.modules] for rt in runtimes]
        assert names == [["person_detection"], [], ["person_detection"]]
        assert runtimes[1].config.enabled_module_names() == []
        # The shared config itself is untouched.
        assert config.enabled_module_names() == ["person_detection"]
        assert run_group(runtimes, RunLimits()) is False
        assert [rt.stats.frames_processed for rt in runtimes] == [15, 15, 15]
