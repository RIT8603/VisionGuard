"""Tests for YouTubeSource and YouTube camera configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend.services.cameras import load_camera_file, parse_cameras, public_view, save_camera_file
from computer_vision.core.config import (
    CameraConfig,
    ConfigError,
    VideoConfig,
    VideoSourceType,
    build_config,
    is_youtube_url,
)
from computer_vision.sources.base import FrameReadError, SourceOpenError
from computer_vision.sources.factory import available_source_types, create_video_source
from computer_vision.sources.youtube import YouTubeSource


class TestYouTubeUrlValidation:
    """Verifies that YouTube URLs are recognized and validated properly."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=M3HKLzjvKPc",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=abc12345",
            "https://youtu.be/M3HKLzjvKPc",
            "https://www.youtube.com/live/M3HKLzjvKPc",
            "https://youtube.com/live/M3HKLzjvKPc?feature=share",
            "https://www.youtube.com/embed/M3HKLzjvKPc",
            "http://www.youtube.com/watch?v=M3HKLzjvKPc",
        ],
    )
    def test_valid_youtube_urls(self, url: str):
        assert is_youtube_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/video.mp4",
            "rtsp://192.168.1.1:554/live",
            "file:///path/to/video.avi",
            "http://vimeo.com/12345",
            "not-a-url",
            "",
            None,
        ],
    )
    def test_invalid_youtube_urls(self, url: str | None):
        assert is_youtube_url(url) is False


class TestYouTubeCameraConfig:
    """Verifies CameraConfig and VideoConfig handling for YouTube sources."""

    def test_valid_youtube_camera_config(self):
        cam = CameraConfig(
            id="yt-cam",
            name="Live Stream",
            source=VideoSourceType.YOUTUBE,
            youtube_url="https://www.youtube.com/watch?v=M3HKLzjvKPc",
        )
        assert cam.id == "yt-cam"
        assert cam.source is VideoSourceType.YOUTUBE
        assert cam.youtube_url == "https://www.youtube.com/watch?v=M3HKLzjvKPc"

        video_cfg = cam.to_video(VideoConfig(source=VideoSourceType.YOUTUBE, youtube_url="https://www.youtube.com/watch?v=M3HKLzjvKPc"))
        assert video_cfg.source is VideoSourceType.YOUTUBE
        assert video_cfg.youtube_url == "https://www.youtube.com/watch?v=M3HKLzjvKPc"
        assert video_cfg.source_id == "yt-cam"

    def test_missing_youtube_url_raises(self):
        with pytest.raises(Exception):
            CameraConfig(
                id="yt-cam",
                source=VideoSourceType.YOUTUBE,
                youtube_url=None,
            )

    def test_invalid_youtube_domain_raises(self):
        with pytest.raises(Exception):
            CameraConfig(
                id="yt-cam",
                source=VideoSourceType.YOUTUBE,
                youtube_url="https://vimeo.com/12345",
            )

    def test_parse_cameras_with_youtube_source(self):
        items = [
            {
                "id": "yt-live",
                "name": "NASA ISS Stream",
                "source": "youtube",
                "youtube_url": "https://www.youtube.com/watch?v=M3HKLzjvKPc",
                "modules": ["person_detection"],
            }
        ]
        cameras, errors = parse_cameras(items, max_cameras=4)
        assert errors == []
        assert len(cameras) == 1
        cam = cameras[0]
        assert cam.id == "yt-live"
        assert cam.source is VideoSourceType.YOUTUBE
        assert cam.youtube_url == "https://www.youtube.com/watch?v=M3HKLzjvKPc"
        assert cam.modules == ["person_detection"]

    def test_public_view_includes_youtube_url(self):
        cam = CameraConfig(
            id="yt-1",
            source=VideoSourceType.YOUTUBE,
            youtube_url="https://www.youtube.com/watch?v=M3HKLzjvKPc",
        )
        view = public_view(cam)
        assert view["source"] == "youtube"
        assert view["youtube_url"] == "https://www.youtube.com/watch?v=M3HKLzjvKPc"

    def test_save_and_load_camera_file_persistence(self, tmp_path):
        cam_file = tmp_path / "cameras.yaml"
        original = [
            CameraConfig(
                id="yt-stream",
                name="Public Broadcast",
                source=VideoSourceType.YOUTUBE,
                youtube_url="https://www.youtube.com/live/M3HKLzjvKPc",
                modules=["object_detection"],
            )
        ]
        save_camera_file(cam_file, original)
        assert cam_file.exists()

        loaded = load_camera_file(cam_file)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].id == "yt-stream"
        assert loaded[0].source is VideoSourceType.YOUTUBE
        assert loaded[0].youtube_url == "https://www.youtube.com/live/M3HKLzjvKPc"
        assert loaded[0].modules == ["object_detection"]


class TestYouTubeSourceLifecycle:
    """Verifies YouTubeSource creation, resolution, frame capture, and errors."""

    def test_factory_builds_youtube_source(self):
        cfg = build_config({
            "video": {
                "source": "youtube",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            }
        }).video
        source = create_video_source(cfg, "yt-0")
        assert isinstance(source, YouTubeSource)
        assert source.source_type == "youtube"
        assert source.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_uninstalled_ytdlp_raises_error(self, monkeypatch):
        cfg = build_config({
            "video": {
                "source": "youtube",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            }
        }).video
        source = YouTubeSource(cfg)

        with patch.dict("sys.modules", {"yt_dlp": None}):
            with pytest.raises(SourceOpenError, match="yt-dlp is not installed"):
                source._resolve_stream_url()

    def test_invalid_url_raises_source_open_error(self):
        cfg = build_config({
            "video": {
                "source": "youtube",
                "youtube_url": "https://example.com/not-youtube",
            }
        }).video
        source = YouTubeSource(cfg)
        with pytest.raises(SourceOpenError, match="Invalid YouTube URL"):
            source._resolve_stream_url()

    def test_successful_resolution_and_reading(self, monkeypatch):
        cfg = build_config({
            "video": {
                "source": "youtube",
                "youtube_url": "https://www.youtube.com/watch?v=M3HKLzjvKPc",
                "read_timeout_sec": 1.0,
            }
        }).video
        source = YouTubeSource(cfg)

        # Mock yt_dlp extraction
        mock_info = {
            "title": "NASA ISS Live Stream",
            "formats": [
                {"format_id": "231", "height": 480, "url": "https://stream.fake/480.m3u8"},
                {"format_id": "311", "height": 720, "url": "https://stream.fake/720.m3u8"},
            ],
        }

        class FakeYDL:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def extract_info(self, url, download=False):
                return mock_info

        import yt_dlp
        monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

        resolved = source._resolve_stream_url()
        assert resolved == "https://stream.fake/720.m3u8"
        assert source.stream_title == "NASA ISS Live Stream"

        # Mock cv2.VideoCapture
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        fake_cap.read.return_value = (True, fake_frame)
        fake_cap.get.side_effect = lambda prop: {3: 1280, 4: 720, 5: 30.0}.get(prop, 0.0)

        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: fake_cap)

        info = source.open()
        assert info.width == 1280
        assert info.height == 720
        assert source.is_open is True

        # Read frame
        packet = source.read()
        assert packet is not None
        assert packet.frame.shape == (720, 1280, 3)
        assert packet.source_id == source.source_id

        # Peek
        peek_frame, seq = source.peek()
        assert peek_frame is not None
        assert seq > 0

        source.release()
        assert source.is_open is False

    def test_reconnect_attempts_exceeded_raises_frame_read_error(self, monkeypatch):
        cfg = build_config({
            "video": {
                "source": "youtube",
                "youtube_url": "https://www.youtube.com/watch?v=M3HKLzjvKPc",
                "read_timeout_sec": 0.01,
                "reconnect_attempts": 2,
                "reconnect_delay_sec": 0.01,
            }
        }).video
        source = YouTubeSource(cfg)
        source._is_open = True
        source._reader = MagicMock()
        source._reader.latest.return_value = None
        source._reader.failed = True

        # First reconnect attempt (fails to reconnect)
        with patch.object(source, "_resolve_stream_url"), patch("computer_vision.sources.opencv_base.OpenCVVideoSource.open", side_effect=SourceOpenError("failed")):
            source.read()
            assert source.is_recovering is True
            assert source.total_reconnects == 1

        # Second reconnect attempt (fails again)
        with patch.object(source, "_resolve_stream_url"), patch("computer_vision.sources.opencv_base.OpenCVVideoSource.open", side_effect=SourceOpenError("failed")):
            source.read()
            assert source.total_reconnects == 2

        # Third attempt exceeds reconnect_attempts (2) -> raises FrameReadError
        with pytest.raises(FrameReadError, match="The stream appears to be offline or finished"):
            source.read()
