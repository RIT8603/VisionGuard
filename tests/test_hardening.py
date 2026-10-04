"""Regression tests for the 2026-09 bug & security audit.

Each class pins one finding so it cannot quietly come back. The numbering
follows the audit report (H = high, M = medium, L = low).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from backend.services.state import DashboardState
from backend.services.web_sink import WebSink
from computer_vision.core.config import ConfigError, build_config
from computer_vision.core.context import BBox, Detection, FrameContext, json_safe
from computer_vision.inference.face_base import RawFace
from computer_vision.utils.timing import PipelineStats

fastapi = pytest.importorskip("fastapi", reason="dashboard extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.server import (  # noqa: E402
    TOKEN_COOKIE,
    AccessPolicy,
    build_app,
)

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "t0ken-for-tests-only-0123456789"


def face_context() -> FrameContext:
    """What face_detection + face_recognition leave in a context."""
    ctx = FrameContext(
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
        frame_id=3,
        source_id="cam",
        monotonic_ts=0.0,
    )
    raw = RawFace(
        confidence=0.97,
        xyxy=(10.0, 10.0, 50.0, 60.0),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        raw_row=np.zeros(15, dtype=np.float32),
    )
    ctx.add_detections(
        "face_detection",
        [Detection("face", np.float32(0.97), BBox(10, 10, 50, 60), extra={"face": raw})],
    )
    ctx.add_detections(
        "face_recognition",
        [
            Detection(
                "alice",
                0.97,
                BBox(10, 10, 50, 60),
                extra={
                    "identity": "alice",
                    "similarity": np.float32(0.81),
                    "recognised": np.bool_(True),
                },
            )
        ],
    )
    ctx.metadata["latency_ms"] = float("nan")
    return ctx


# --------------------------------------------------------------------------- #
# H1 - /api/status must survive NumPy / backend objects in results
# --------------------------------------------------------------------------- #

class TestH1JsonSafeStatus:
    def test_summary_is_strict_json(self):
        summary = face_context().summary()
        text = json.dumps(summary, allow_nan=False)  # what Starlette does
        assert "RawFace" not in text

    def test_numpy_values_are_kept_as_plain_numbers(self):
        summary = face_context().summary()
        rec = summary["detections"]["face_recognition"][0]
        assert rec["extra"]["similarity"] == pytest.approx(0.81, abs=1e-6)
        assert rec["extra"]["recognised"] is True
        assert summary["detections"]["face_detection"][0]["confidence"] == pytest.approx(0.97)

    def test_backend_object_is_dropped_not_fatal(self):
        det = face_context().summary()["detections"]["face_detection"][0]
        assert "face" not in det.get("extra", {})

    def test_nan_becomes_null(self):
        assert face_context().summary()["metadata"]["latency_ms"] is None
        assert json_safe({"x": math.inf}) == {"x": None}

    def test_api_status_returns_200_with_faces(self):
        state = DashboardState()
        state.info = {"modules": []}
        config = build_config({"processing": {"show_window": False}}).processing
        WebSink(state, config, stream_fps=0).emit(face_context(), PipelineStats())

        client = TestClient(build_app(state), base_url="http://127.0.0.1")
        response = client.get("/api/status")

        assert response.status_code == 200
        assert response.json()["frame"]["frame_id"] == 3


# --------------------------------------------------------------------------- #
# H2 - no server value may be parsed as HTML by the dashboard
# --------------------------------------------------------------------------- #

class TestH2NoHtmlInjection:
    html = (ROOT / "frontend" / "static" / "index.html").read_text(encoding="utf-8")

    def _script(self) -> str:
        return self.html.split("<script>", 1)[1].split("</script>", 1)[0]

    def test_every_innerhtml_interpolation_is_escaped(self):
        script = self._script()
        # Every ${...} inside an innerHTML template must go through esc(),
        # num() or cls().
        for block in re.findall(r"innerHTML\s*=\s*(.*?);\n", script, re.S):
            for expr in re.findall(r"\$\{([^}]*)\}", block):
                assert re.match(r"\s*(esc|num|cls)\(", expr), expr

    def test_events_are_built_with_text_nodes(self):
        script = self._script()
        add_events = script.split("function addEvents", 1)[1]
        assert "innerHTML" not in add_events
        assert "textContent" in add_events

    def test_security_headers_are_sent(self):
        client = TestClient(build_app(DashboardState()), base_url="http://127.0.0.1")
        headers = client.get("/").headers
        assert "default-src 'none'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"


# --------------------------------------------------------------------------- #
# H3 - Host check (DNS rebinding) and token authentication
# --------------------------------------------------------------------------- #

class TestH3HostCheck:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
    def test_loopback_names_are_accepted(self, host):
        client = TestClient(build_app(DashboardState()), base_url=f"http://{host}:8080")
        assert client.get("/api/status").status_code == 200

    @pytest.mark.parametrize("host", ["evil.example", "attacker.test:8080", "192.168.1.9"])
    def test_foreign_host_header_is_rejected(self, host):
        """A DNS-rebound page sends its own name in Host."""
        client = TestClient(build_app(DashboardState()), base_url="http://127.0.0.1")
        for path in ("/api/status", "/api/snapshot.jpg", "/stream.mjpg", "/"):
            assert client.get(path, headers={"host": host}).status_code == 400

    def test_extra_allowed_host(self):
        policy = AccessPolicy.for_bind("127.0.0.1", extra_hosts=["guard.lan"])
        client = TestClient(build_app(DashboardState(), policy=policy), base_url="http://guard.lan")
        assert client.get("/api/status").status_code == 200


class TestH3Token:
    def _client(self, host="127.0.0.1", token=TOKEN, base="http://127.0.0.1"):
        policy = AccessPolicy.for_bind(host, token=token)
        return TestClient(build_app(DashboardState(), policy=policy), base_url=base)

    def test_localhost_without_token_still_works(self):
        """Existing default behaviour is unchanged."""
        policy = AccessPolicy.for_bind("127.0.0.1")
        assert policy.token is None
        client = TestClient(build_app(DashboardState(), policy=policy), base_url="http://127.0.0.1")
        assert client.get("/api/status").status_code == 200

    def test_missing_or_wrong_token_is_401(self):
        client = self._client()
        assert client.get("/api/status").status_code == 401
        assert client.get("/stream.mjpg").status_code == 401
        assert client.get("/api/snapshot.jpg?token=wrong").status_code == 401
        assert client.get(
            "/api/status", headers={"authorization": "Bearer nope"}
        ).status_code == 401

    def test_bearer_token(self):
        response = self._client().get(
            "/api/status", headers={"authorization": f"Bearer {TOKEN}"}
        )
        assert response.status_code == 200

    def test_query_token_sets_httponly_cookie_for_the_browser(self):
        client = self._client()
        first = client.get(f"/?token={TOKEN}")
        assert first.status_code == 200
        cookie = first.headers["set-cookie"]
        assert cookie.startswith(f"{TOKEN_COOKIE}=")
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
        # The cookie alone now authorises the page's own fetch() / <img>.
        assert client.get("/api/status").status_code == 200

    def test_network_bind_always_requires_a_token(self):
        policy = AccessPolicy.for_bind("0.0.0.0")
        assert policy.token and len(policy.token) >= 24
        assert policy.token_generated is True
        # Any Host is allowed on a wildcard bind (the token is the guard)...
        assert policy.allowed_hosts == frozenset()
        client = TestClient(build_app(DashboardState(), policy=policy), base_url="http://10.0.0.8")
        assert client.get("/api/status").status_code == 401
        assert client.get(
            "/api/status", headers={"authorization": f"Bearer {policy.token}"}
        ).status_code == 200

    def test_generated_tokens_are_unique(self):
        assert AccessPolicy.for_bind("0.0.0.0").token != AccessPolicy.for_bind("0.0.0.0").token

    def test_configured_token_is_used_and_not_regenerated(self):
        policy = AccessPolicy.for_bind("0.0.0.0", token=TOKEN)
        assert policy.token == TOKEN and policy.token_generated is False

    def test_token_config_from_env_and_validation(self, monkeypatch):
        from computer_vision.core.config import apply_env_overrides

        monkeypatch.setenv("VISIONGUARD_DASHBOARD_TOKEN", TOKEN)
        config = build_config(apply_env_overrides({}))
        assert config.dashboard.auth_token.get_secret_value() == TOKEN
        assert TOKEN not in repr(config)  # SecretStr keeps it out of logs

        with pytest.raises(ConfigError, match="auth_token"):
            build_config({"dashboard": {"auth_token": "short"}})
        assert build_config({"dashboard": {"auth_token": ""}}).dashboard.auth_token is None


# --------------------------------------------------------------------------- #
# M4 - a busy port must reach the caller
# --------------------------------------------------------------------------- #

import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from backend.api.server import DashboardServer  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestM4PortInUse:
    def test_busy_port_raises_oserror_in_the_caller(self):
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        try:
            server = DashboardServer(DashboardState(), "127.0.0.1", port)
            with pytest.raises(OSError):
                server.start()
        finally:
            blocker.close()

    def test_server_really_serves_after_start(self):
        import http.client

        port = _free_port()
        state = DashboardState()
        server = DashboardServer(state, "127.0.0.1", port)
        server.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/api/status")
            assert conn.getresponse().status == 200
        finally:
            server.stop()

    def test_main_reports_busy_port_with_exit_code(self, tmp_path, capsys):
        from backend import main as entry

        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "video: {source: file, path: nope.mp4}\n"
            "processing: {show_window: false}\n"
            f"dashboard: {{enabled: true, port: {port}}}\n",
            encoding="utf-8",
        )
        try:
            code = entry.main(["--config", str(cfg)])
        finally:
            blocker.close()
        assert code == entry.EXIT_CONFIG
        assert "Could not bind" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# M5 - live streams must not exhaust the server
# --------------------------------------------------------------------------- #

class TestM5StreamLimits:
    def _open_stream(self, port):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(b"GET /stream.mjpg HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        return sock, sock.recv(64)

    def test_extra_viewers_get_503_and_api_stays_responsive(self):
        import http.client

        port = _free_port()
        state = DashboardState()
        state.publish(b"\xff\xd8frame", {}, {}, [])
        # Start with a high cap to show streams no longer tie up threads.
        server = DashboardServer(state, "127.0.0.1", port, max_streams=60)
        server.start()
        socks = []
        try:
            for _ in range(50):  # more than the 40-thread pool that used to starve
                sock, head = self._open_stream(port)
                assert head.startswith(b"HTTP/1.1 200")
                socks.append(sock)

            started = time.perf_counter()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/api/status")
            assert conn.getresponse().status == 200
            assert time.perf_counter() - started < 2.0
        finally:
            for sock in socks:
                sock.close()
            server.stop()

    def test_cap_rejects_the_viewer_over_the_limit_and_frees_slots(self):
        port = _free_port()
        state = DashboardState()
        state.publish(b"\xff\xd8frame", {}, {}, [])
        server = DashboardServer(state, "127.0.0.1", port, max_streams=2)
        server.start()
        try:
            a, head_a = self._open_stream(port)
            b, head_b = self._open_stream(port)
            assert head_a.startswith(b"HTTP/1.1 200") and head_b.startswith(b"HTTP/1.1 200")
            c, head_c = self._open_stream(port)
            assert head_c.startswith(b"HTTP/1.1 503")
            c.close()

            a.close()  # a viewer leaves: its slot must come back
            state.publish(b"\xff\xd8next", {}, {}, [])  # disconnect noticed on write
            deadline = time.time() + 5
            while time.time() < deadline:
                d, head_d = self._open_stream(port)
                d.close()
                if head_d.startswith(b"HTTP/1.1 200"):
                    break
                state.publish(b"\xff\xd8more", {}, {}, [])
                time.sleep(0.1)
            assert head_d.startswith(b"HTTP/1.1 200")
            b.close()
        finally:
            server.stop()

    def test_stream_generator_uses_no_worker_thread(self):
        import inspect

        from backend.api import server as server_module

        assert inspect.isasyncgenfunction(server_module._mjpeg_frames)


# --------------------------------------------------------------------------- #
# M6 / M7 - RTSP recovery vs stall timeout, reader/release race
# --------------------------------------------------------------------------- #

import computer_vision.sources.opencv_base as opencv_base  # noqa: E402
from computer_vision.sources.base import BaseVideoSource, FramePacket, SourceInfo  # noqa: E402
from computer_vision.sources.rtsp import RTSPSource, _LatestFrameReader  # noqa: E402


class TestM6RecoveryIsNotAStall:
    def test_pipeline_waits_while_the_source_recovers(self):
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import RunLimits, VisionPipeline
        from tests.test_pipeline import RecordingSink

        class Recovering(BaseVideoSource):
            """No frames for 0.5 s (a reconnect), then frames again."""

            source_type = "fake"

            def __init__(self):
                super().__init__("fake")
                self.t0 = None

            def open(self):
                self._is_open = True
                self.t0 = time.perf_counter()
                return self.info

            @property
            def info(self):
                return SourceInfo("fake", "fake", 32, 24, 10.0)

            @property
            def is_recovering(self):
                return time.perf_counter() - self.t0 < 0.5

            def read(self):
                time.sleep(0.01)
                if self.is_recovering:
                    return None
                return FramePacket(np.zeros((24, 32, 3), np.uint8), self._next_frame_id(), "fake")

            def release(self):
                self._is_open = False

        config = build_config(
            {"processing": {"show_window": False, "target_width": None, "stall_timeout_sec": 0.2}}
        )
        stats = VisionPipeline(
            config=config,
            source=Recovering(),
            modules=[],
            output=RecordingSink(),
            processor=FrameProcessor(config.processing),
        ).run(RunLimits(max_frames=3, max_seconds=5))

        # Before the fix the 0.2 s stall timeout ended the run at 0 frames.
        assert stats.frames_processed == 3

    def test_rtsp_reports_recovering_during_reconnect(self, monkeypatch):
        config = build_config(
            {
                "video": {
                    "source": "rtsp",
                    "rtsp_url": "rtsp://cam/stream",
                    "reconnect_delay_sec": 0.0,
                    "read_timeout_sec": 0.05,
                }
            }
        ).video

        opened = {"ok": False}

        class Cap:
            def __init__(self, *a):
                self._ok = opened["ok"]

            def isOpened(self):  # noqa: N802
                return self._ok

            def read(self):
                time.sleep(0.005)
                return False, None

            def get(self, prop):
                return 0.0

            def set(self, *a):
                return True

            def release(self):
                pass

        monkeypatch.setattr(opencv_base.cv2, "VideoCapture", Cap)
        source = RTSPSource(config)
        opened["ok"] = True
        source.open()
        opened["ok"] = False  # the camera is now down
        deadline = time.time() + 2
        while not source.is_recovering and time.time() < deadline:
            source.read()
        assert source.is_recovering is True
        source.release()
        assert source.is_recovering is False


class TestM7ReaderNeverRacesRelease:
    class BlockingCapture:
        """read() blocks until released by the test; release() is recorded."""

        def __init__(self):
            self.gate = threading.Event()
            self.in_read = threading.Event()
            self.release_during_read = False
            self.released = False

        def read(self):
            self.in_read.set()
            self.gate.wait(5)
            self.in_read.clear()
            return False, None

        def release(self):
            if self.in_read.is_set():
                self.release_during_read = True
            self.released = True

    def test_stuck_reader_takes_ownership_of_release(self):
        capture = self.BlockingCapture()
        reader = _LatestFrameReader(capture, "t", join_timeout=0.1)
        reader.start()
        assert capture.in_read.wait(1)

        safe = reader.stop()  # join times out: the thread is inside read()

        assert safe is False
        assert capture.released is False  # nobody released it under the read
        capture.gate.set()  # the read returns...
        deadline = time.time() + 2
        while not capture.released and time.time() < deadline:
            time.sleep(0.01)
        assert capture.released is True  # ...and the thread released it
        assert capture.release_during_read is False

    def test_normal_stop_lets_the_caller_release(self):
        capture = self.BlockingCapture()
        capture.gate.set()  # reads return immediately (and fail)
        reader = _LatestFrameReader(capture, "t", join_timeout=1.0)
        reader.start()
        time.sleep(0.05)
        assert reader.stop() is True
        assert capture.released is False  # caller's job in this case

    def test_join_timeout_exceeds_read_timeout(self, monkeypatch):
        config = build_config(
            {"video": {"source": "rtsp", "rtsp_url": "rtsp://cam/s", "read_timeout_sec": 5.0}}
        ).video
        source = RTSPSource(config)
        source._capture = self.BlockingCapture()
        source._capture.gate.set()
        source._start_reader()
        assert source._reader._join_timeout > config.read_timeout_sec
        source.release()


# --------------------------------------------------------------------------- #
# M8 - the event feed must not lose events in a burst
# --------------------------------------------------------------------------- #

class TestM8EventPaging:
    def test_client_paging_sees_every_event_of_a_burst(self):
        state = DashboardState(max_events=500)
        state.publish(None, {}, {}, [{"event": f"e{i}"} for i in range(1, 131)])
        client = TestClient(build_app(state), base_url="http://127.0.0.1")

        seen, since = [], 0
        for _ in range(10):  # what the dashboard's pollEvents() does
            body = client.get(f"/api/events?since={since}&limit=50").json()
            page = body["events"]
            assert page == sorted(page, key=lambda e: -e["id"])  # newest first
            seen.extend(e["id"] for e in page)
            if page:
                since = max(e["id"] for e in page)
            if len(page) < 50:
                break

        assert sorted(seen) == list(range(1, 131))
        assert body["last_id"] == 130

    def test_last_id_lets_the_client_detect_a_restart(self):
        client = TestClient(build_app(DashboardState()), base_url="http://127.0.0.1")
        assert client.get("/api/events?since=999").json() == {"events": [], "last_id": 0}

    def test_frontend_pages_and_resets(self):
        script = (ROOT / "frontend" / "static" / "index.html").read_text(encoding="utf-8")
        assert "last_id < lastEventId" in script
        assert "events.length < EVENT_PAGE" in script


# --------------------------------------------------------------------------- #
# M9 - model files are verified before the pickle loader sees them
# --------------------------------------------------------------------------- #

from computer_vision.inference.base import ModelLoadError, verify_model_file  # noqa: E402


class TestM9ModelIntegrity:
    def test_missing_model_is_not_downloaded_implicitly(self, tmp_path):
        called = []
        with pytest.raises(ModelLoadError, match="does not download"):
            verify_model_file(str(tmp_path / "m.pt"), download=called.append)
        assert called == []

    def test_hash_mismatch_is_refused(self, tmp_path):
        weights = tmp_path / "m.pt"
        weights.write_bytes(b"not the real weights")
        with pytest.raises(ModelLoadError, match="integrity"):
            verify_model_file(str(weights), expected_sha256="0" * 64)

    def test_matching_hash_passes(self, tmp_path):
        import hashlib

        weights = tmp_path / "m.pt"
        weights.write_bytes(b"weights")
        digest = hashlib.sha256(b"weights").hexdigest()
        assert verify_model_file(str(weights), expected_sha256=digest.upper()) == str(weights)

    def test_allowed_download_is_verified_before_use(self, tmp_path):
        target = tmp_path / "m.pt"

        def fake_download(ref):
            Path(ref).write_bytes(b"tampered")
            return ref

        with pytest.raises(ModelLoadError, match="integrity"):
            verify_model_file(
                str(target), expected_sha256="0" * 64, allow_download=True, download=fake_download
            )

    def test_backends_refuse_before_constructing_yolo(self, tmp_path, monkeypatch):
        import sys
        import types

        constructed = []
        fake = types.ModuleType("ultralytics")
        fake.YOLO = lambda *a, **k: constructed.append(a)
        monkeypatch.setitem(sys.modules, "ultralytics", fake)

        from computer_vision.inference.pose_backend import UltralyticsPoseBackend
        from computer_vision.inference.ultralytics_backend import UltralyticsDetector

        bad = tmp_path / "evil.pt"
        bad.write_bytes(b"pickle payload")
        for backend in (
            UltralyticsDetector({"model": str(bad), "model_sha256": "1" * 64}),
            UltralyticsPoseBackend({"model": str(bad), "model_sha256": "1" * 64}),
            UltralyticsDetector({"model": str(tmp_path / "missing.pt")}),
        ):
            with pytest.raises(ModelLoadError):
                backend.load()
        assert constructed == []

    def test_requirements_pin_ultralytics_exactly(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        line = next(l for l in text.splitlines() if l.startswith("ultralytics"))
        assert "==" in line
        version = line.split("==", 1)[1].split()[0]
        assert version not in {"8.3.41", "8.3.42", "8.3.45", "8.3.46"}

    def test_config_pins_bundled_weights(self):
        import yaml

        raw = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        for name in ("person_detection", "pose_detection"):
            assert len(raw["modules"][name]["model_sha256"]) == 64


# --------------------------------------------------------------------------- #
# L10 - an open failure must not leak the capture, the sink or the modules
# --------------------------------------------------------------------------- #

class TestL10OpenFailureCleanup:
    def test_configure_failure_releases_the_capture(self, monkeypatch, tmp_path):
        from computer_vision.sources.base import SourceOpenError
        from computer_vision.sources.video_file import VideoFileSource

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        released = []

        class Cap:
            def __init__(self, *a):
                pass

            def isOpened(self):  # noqa: N802
                return True

            def get(self, prop):
                return 10.0  # 10 frames

            def set(self, *a):
                return True

            def release(self):
                released.append(True)

        monkeypatch.setattr(opencv_base.cv2, "VideoCapture", Cap)
        cfg = build_config({"video": {"source": "file", "path": str(video), "start_frame": 99}}).video
        source = VideoFileSource(cfg)
        with pytest.raises(SourceOpenError, match="Cannot seek"):
            source.open()
        assert released == [True]
        assert source._capture is None and not source.is_open

    def test_pipeline_shuts_down_when_open_fails(self):
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.output import NullSink
        from computer_vision.core.pipeline import VisionPipeline
        from computer_vision.sources.base import SourceOpenError

        calls = []

        class Broken(BaseVideoSource):
            def open(self):
                raise SourceOpenError("camera missing")

            def read(self):  # pragma: no cover
                return None

            def release(self):
                calls.append("release")

            @property
            def info(self):  # pragma: no cover
                raise AssertionError

        class Sink(NullSink):
            def close(self):
                calls.append("sink")

        class Mod:
            module_name = "m"

            def cleanup(self):
                calls.append("module")

        config = build_config({"processing": {"show_window": False}})
        with pytest.raises(SourceOpenError):
            VisionPipeline(
                config=config,
                source=Broken("x"),
                modules=[Mod()],
                output=Sink(log_every=0),
                processor=FrameProcessor(config.processing),
            ).run()
        assert calls == ["module", "sink", "release"]


# --------------------------------------------------------------------------- #
# L11 - RTSP secrets in the query string, and on the command line
# --------------------------------------------------------------------------- #

from computer_vision.sources.rtsp import redact_url  # noqa: E402


class TestL11Redaction:
    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://cam/stream?user=admin&password=s3cret&channel=1",
            "rtsp://cam/stream?pwd=s3cret",
            "rtsp://admin:s3cret@cam/stream?token=s3cret",
            "rtsp://admin:s3cret@cam:99999/stream?auth=s3cret",  # bad port
            "rtsp://admin:s3cret@[fe80::1]:554/stream",
        ],
    )
    def test_no_secret_survives(self, url):
        assert "s3cret" not in redact_url(url)

    def test_non_secret_query_is_kept(self):
        assert redact_url("rtsp://cam/s?channel=1&subtype=0") == "rtsp://cam/s?channel=1&subtype=0"

    def test_cli_password_triggers_a_warning(self, caplog):
        import argparse
        import logging

        from backend.main import _warn_if_cli_secret

        with caplog.at_level(logging.WARNING):
            _warn_if_cli_secret(argparse.Namespace(rtsp_url="rtsp://a:b@c/d"))
        assert "shell history" in caplog.text
        assert "a:b@" not in caplog.text


# --------------------------------------------------------------------------- #
# L12 - identity store integrity
# --------------------------------------------------------------------------- #

from computer_vision.identity.store import IdentityStore, IdentityStoreError  # noqa: E402


class TestL12IdentityStore:
    def _store(self):
        store = IdentityStore()
        store.enroll("alice", np.ones(8, dtype=np.float32))
        store.enroll("bob", np.arange(8, dtype=np.float32))
        return store

    def test_saves_to_the_exact_path_given(self, tmp_path):
        path = self._store().save(tmp_path / "ids.store")  # no .npz suffix
        assert path == tmp_path / "ids.store" and path.is_file()
        assert IdentityStore.load(path).names == ["alice", "bob"]
        assert not list(tmp_path.glob("*.tmp"))

    def test_mismatched_arrays_are_rejected_cleanly(self, tmp_path):
        bad = tmp_path / "bad.npz"
        np.savez(
            bad,
            version=np.array([1], dtype=np.int32),
            names=np.array(["a", "b", "c"]),
            enrolled_at=np.array(["t", "t", "t"]),
            embeddings=np.ones((1, 8), dtype=np.float32),
        )
        with pytest.raises(IdentityStoreError, match="malformed"):
            IdentityStore.load(bad)

    def test_nan_embeddings_are_rejected(self, tmp_path):
        bad = tmp_path / "nan.npz"
        emb = np.ones((1, 8), dtype=np.float32)
        emb[0, 0] = np.nan
        np.savez(
            bad,
            version=np.array([1], dtype=np.int32),
            names=np.array(["a"]),
            enrolled_at=np.array(["t"]),
            embeddings=emb,
        )
        with pytest.raises(IdentityStoreError, match="NaN"):
            IdentityStore.load(bad)

    def test_failed_save_keeps_the_previous_store(self, tmp_path, monkeypatch):
        path = self._store().save(tmp_path / "ids.npz")
        before = path.read_bytes()

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(np, "savez_compressed", boom)
        with pytest.raises(OSError):
            IdentityStore().save(path)
        assert path.read_bytes() == before
        assert not list(tmp_path.glob(".*.tmp"))

    @pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX permissions")
    def test_store_is_owner_only(self, tmp_path):
        path = self._store().save(tmp_path / "ids.npz")
        assert (path.stat().st_mode & 0o077) == 0


# --------------------------------------------------------------------------- #
# L13 - nonsensical config values are rejected at load time
# --------------------------------------------------------------------------- #

class TestL13ConfigValidation:
    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("dashboard", "max_events", 0),
            ("dashboard", "max_events", -5),
            ("dashboard", "max_streams", 0),
            ("video", "reconnect_attempts", -1),
            ("video", "reconnect_delay_sec", -2.0),
            ("video", "read_timeout_sec", -1),
            ("video", "open_timeout_sec", -1),
            ("video", "read_retries", -1),
            ("processing", "hud_max_detections", -1),
            ("processing", "stall_timeout_sec", -1),
            ("processing", "max_fps", -1),
            ("processing", "log_stats_every", -1),
        ],
    )
    def test_rejected(self, section, key, value):
        with pytest.raises(ConfigError, match=key):
            build_config({section: {key: value}})

    def test_zero_keeps_its_documented_meaning(self):
        config = build_config(
            {"video": {"reconnect_attempts": 0}, "processing": {"stall_timeout_sec": 0, "max_fps": 0}}
        )
        assert config.video.reconnect_attempts == 0

    def test_shipped_config_is_valid(self):
        from computer_vision.core.config import load_config

        load_config(ROOT / "configs" / "config.yaml", use_env=False)


# --------------------------------------------------------------------------- #
# L14 - fall detection state is bounded
# --------------------------------------------------------------------------- #

class TestL14FallStateBounded:
    def test_cooldown_entries_are_dropped_with_their_tracks(self):
        from computer_vision.core.context import Pose
        from computer_vision.modules.analysis.fall import FallDetectionModule

        module = FallDetectionModule({"min_frames": 1, "max_age": 1})
        module.initialize()
        # A lying (wide) box somewhere new each time: a new track each time.
        for i in range(60):
            ctx = FrameContext(
                frame=np.zeros((480, 640, 3), np.uint8), frame_id=i, source_id="c",
                monotonic_ts=float(i),
            )
            x = (i % 6) * 100
            ctx.add_poses("pose_detection", [Pose(bbox=BBox(x, 400, x + 90, 430), confidence=0.9)])
            module.process(ctx)
            for _ in range(3):  # let the track expire
                empty = FrameContext(
                    frame=ctx.frame, frame_id=i, source_id="c", monotonic_ts=float(i)
                )
                module.process(empty)

        assert module.total_falls >= 50
        assert len(module._reported) <= 2  # was: one entry per fall, forever


# --------------------------------------------------------------------------- #
# L15 - "Press q" only when a window really exists
# --------------------------------------------------------------------------- #

class TestL15WindowHint:
    def test_has_window(self):
        from backend.main import _has_window
        from computer_vision.core.output import MultiSink, NullSink, WindowSink

        processing = build_config({}).processing
        window = WindowSink(processing, "a", "b", [])
        web = WebSink(DashboardState(), processing)

        assert _has_window(window)
        assert _has_window(MultiSink([window, web]))
        assert not _has_window(NullSink())
        assert not _has_window(web)  # headless fallback + dashboard
        assert not _has_window(MultiSink([NullSink(), web]))
