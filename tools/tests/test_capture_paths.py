"""The three ways video gets in: an uploaded file, an RTSP URL, a local device.

Covers what the dashboard's camera page relies on beyond the sources
themselves: uploads landing safely, a dashboard file camera playing in real
time, and an RTSP failure being explained instead of just "could not open".
"""

from __future__ import annotations

import asyncio
import socket
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.server import build_app
from backend.services import uploads
from backend.services.camera_manager import CameraManager
from backend.services.hub import CameraHub
from backend.services.reload import ReloadController
from backend.services.uploads import UploadError, safe_video_name, save_upload
from computer_vision.core.config import CameraConfig, VideoConfig
from computer_vision.sources.rtsp_probe import probe_rtsp


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #


class TestSafeVideoName:
    @pytest.mark.parametrize(
        "given, expected",
        [
            ("campus.mp4", "campus.mp4"),
            ("../../etc/evil.MP4", "evil.mp4"),
            ("C:\\Users\\x\\Traffic Cam.mkv", "Traffic Cam.mkv"),
            ("we$ird*na:me.avi", "we_ird_na_me.avi"),
        ],
    )
    def test_only_a_plain_name_is_kept(self, given, expected):
        assert safe_video_name(given) == expected

    @pytest.mark.parametrize("given", ["notes.txt", "run.py", "noext", "", None])
    def test_non_videos_are_refused(self, given):
        with pytest.raises(UploadError):
            safe_video_name(given)


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


class TestSaveUpload:
    def test_saves_and_never_overwrites(self, tmp_path):
        first = asyncio.run(save_upload("a.mp4", _chunks(b"12", b"34"), directory=tmp_path))
        second = asyncio.run(save_upload("a.mp4", _chunks(b"x"), directory=tmp_path))
        assert Path(first["path"]).read_bytes() == b"1234" and first["size"] == 4
        assert Path(second["path"]).name == "a-2.mp4"

    def test_too_large_leaves_nothing_behind(self, tmp_path):
        with pytest.raises(UploadError):
            asyncio.run(save_upload("a.mp4", _chunks(b"x" * 10), directory=tmp_path, max_bytes=5))
        assert list(tmp_path.iterdir()) == []

    def test_empty_is_refused(self, tmp_path):
        with pytest.raises(UploadError):
            asyncio.run(save_upload("a.mp4", _chunks(), directory=tmp_path))


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "VIDEO_DIR", tmp_path)
    monkeypatch.setattr(uploads.save_upload, "__kwdefaults__", {"directory": tmp_path, "max_bytes": 1024})
    monkeypatch.setattr(uploads.list_videos, "__defaults__", (tmp_path,))
    hub = CameraHub()
    hub.configure([("a", "A", "file")])
    reload = ReloadController()
    manager = CameraManager(hub, reload, max_cameras=4, editable=True)
    app = build_app(hub.get(None), hub=hub, cameras=manager, reload=reload)
    return TestClient(app, base_url="http://127.0.0.1"), manager, tmp_path


class TestUploadApi:
    def test_upload_then_listed(self, upload_client):
        client, _, folder = upload_client
        response = client.post("/api/videos?name=gate.mp4", content=b"\x00\x01video")
        assert response.status_code == 201
        assert response.json()["name"] == "gate.mp4"
        assert (folder / "gate.mp4").read_bytes() == b"\x00\x01video"
        listed = client.get("/api/videos").json()["videos"]
        assert [v["name"] for v in listed] == ["gate.mp4"]

    def test_non_video_is_400(self, upload_client):
        client, _, folder = upload_client
        assert client.post("/api/videos?name=x.py", content=b"print(1)").status_code == 400
        assert list(folder.iterdir()) == []

    def test_refused_when_cameras_cannot_be_edited(self, upload_client):
        client, manager, _ = upload_client
        manager._editable = False
        assert client.post("/api/videos?name=a.mp4", content=b"x").status_code == 403


# --------------------------------------------------------------------------- #
# File cameras play in real time
# --------------------------------------------------------------------------- #


class TestFileCameraSpeed:
    def test_benchmark_speed_becomes_real_time(self):
        base = VideoConfig(playback_speed=0.0)
        cam = CameraConfig(id="f", source="file", path="clip.mp4")
        assert cam.to_video(base).playback_speed == 1.0

    def test_an_explicit_speed_is_kept(self):
        base = VideoConfig(playback_speed=2.0)
        cam = CameraConfig(id="f", source="file", path="clip.mp4")
        assert cam.to_video(base).playback_speed == 2.0


# --------------------------------------------------------------------------- #
# RTSP probe
# --------------------------------------------------------------------------- #


def _rtsp_server(reply: bytes):
    """A one-shot fake RTSP server answering ``reply``; returns its port."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.recv(4096)
            conn.sendall(reply)
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    return port


class TestRtspProbe:
    def test_ok_means_nothing_to_report(self):
        port = _rtsp_server(b"RTSP/1.0 200 OK\r\nCSeq: 1\r\n\r\n")
        assert probe_rtsp(f"rtsp://127.0.0.1:{port}/live", timeout=2) is None

    @pytest.mark.parametrize(
        "code, words",
        [(401, "password"), (403, "expired"), (404, "no stream at this path"), (453, "connections")],
    )
    def test_status_codes_are_explained(self, code, words):
        port = _rtsp_server(f"RTSP/1.0 {code} X\r\nCSeq: 1\r\n\r\n".encode())
        reason = probe_rtsp(f"rtsp://127.0.0.1:{port}/live?token=s3cret", timeout=2)
        assert f"RTSP {code}" in reason and words in reason
        assert "s3cret" not in reason

    def test_closed_port(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        assert "refused" in probe_rtsp(f"rtsp://127.0.0.1:{port}/x", timeout=2)

    def test_not_rtsp(self):
        port = _rtsp_server(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        assert "not an RTSP server" in probe_rtsp(f"rtsp://127.0.0.1:{port}/x", timeout=2)

    def test_unknown_host(self):
        assert "could not be found" in probe_rtsp("rtsp://no-such-host.invalid/x", timeout=2)
