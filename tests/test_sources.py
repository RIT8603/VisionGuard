"""Video source abstraction.

The webcam source is exercised against a fake ``cv2.VideoCapture`` so the
suite runs on machines with no camera.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.config import CaptureBackend, VideoConfig, VideoSourceType, build_config
from app.sources import opencv_base
from app.sources.base import (
    BaseVideoSource,
    FramePacket,
    FrameReadError,
    SourceNotOpenError,
    SourceOpenError,
)
from app.sources.factory import available_source_types, create_video_source
from app.sources.rtsp import RTSPSource
from app.sources.video_file import VideoFileSource
from app.sources.webcam import WebcamSource


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    instances: list["FakeCapture"] = []

    def __init__(
        self,
        target,
        backend=0,
        *,
        opened=True,
        frames=None,
        fail_reads=0,
        ignore_set=False,
    ):
        self.target = target
        self.backend = backend
        self._opened = opened
        self._frames_left = frames
        self._fail_reads = fail_reads
        # When True the fake behaves like a driver that refuses mode changes.
        self._ignore_set = ignore_set
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
        }
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def read(self):
        if self._fail_reads > 0:
            self._fail_reads -= 1
            return False, None
        if self._frames_left is not None:
            if self._frames_left <= 0:
                return False, None
            self._frames_left -= 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        if self._ignore_set:
            return False
        self.props[prop] = float(value)
        return True

    def release(self):
        self.released = True
        self._opened = False


@pytest.fixture
def fake_capture(monkeypatch):
    """Patch ``cv2.VideoCapture`` as seen by the OpenCV source base class."""
    FakeCapture.instances.clear()
    created = {}

    def factory(target, backend=0):
        capture = FakeCapture(target, backend, **created.get("kwargs", {}))
        return capture

    monkeypatch.setattr(opencv_base.cv2, "VideoCapture", factory)
    return created


def webcam_config(**kwargs) -> VideoConfig:
    return build_config({"video": {"source": "webcam", **kwargs}}).video


class TestFactory:
    def test_webcam_is_available(self):
        source = create_video_source(webcam_config(device_id=1))
        assert isinstance(source, WebcamSource)
        assert source.source_id == "webcam-1"

    def test_file_source_is_available(self):
        config = build_config({"video": {"source": "file", "path": "x.mp4"}}).video
        assert isinstance(create_video_source(config), VideoFileSource)

    def test_rtsp_source_is_available(self):
        config = build_config({"video": {"source": "rtsp", "rtsp_url": "rtsp://x"}}).video
        assert isinstance(create_video_source(config), RTSPSource)

    def test_every_source_type_is_now_implemented(self):
        assert available_source_types() == {"webcam": True, "file": True, "rtsp": True}

    def test_declared_classes_share_the_abstraction(self):
        """All three sources implement the same contract, today."""
        for cls in (WebcamSource, VideoFileSource, RTSPSource):
            assert issubclass(cls, BaseVideoSource)


class TestWebcamSource:
    def test_open_reports_granted_properties(self, fake_capture):
        source = WebcamSource(webcam_config(width=1280, height=720))
        info = source.open()

        assert source.is_open
        assert info.source_id == "webcam-0"
        assert info.source_type == "webcam"
        assert (info.width, info.height) == (1280, 720)
        source.release()

    def test_driver_refusing_the_requested_mode_is_reported(self, fake_capture, caplog):
        """A real driver often snaps to its own nearest mode; report, don't assume."""
        fake_capture["kwargs"] = {"ignore_set": True}
        source = WebcamSource(webcam_config(width=1920, height=1080))

        with caplog.at_level("WARNING"):
            info = source.open()

        assert (info.width, info.height) == (640, 480)  # what the driver granted
        assert "did not grant 1920x1080" in caplog.text
        source.release()

    def test_requested_properties_are_applied(self, fake_capture):
        source = WebcamSource(webcam_config(width=800, height=600, fps=25))
        source.open()

        capture = FakeCapture.instances[-1]
        assert capture.props[cv2.CAP_PROP_FPS] == 25.0
        assert capture.props[cv2.CAP_PROP_BUFFERSIZE] == 1.0  # low-latency capture
        source.release()

    def test_read_produces_frame_packets_with_increasing_ids(self, fake_capture):
        source = WebcamSource(webcam_config())
        source.open()

        first, second = source.read(), source.read()

        assert isinstance(first, FramePacket)
        assert (first.frame_id, second.frame_id) == (1, 2)
        assert first.source_id == "webcam-0"
        assert (first.width, first.height) == (640, 480)
        assert first.capture_ms >= 0.0
        source.release()

    def test_read_before_open_is_rejected(self, fake_capture):
        with pytest.raises(SourceNotOpenError):
            WebcamSource(webcam_config()).read()

    def test_open_failure_message_is_actionable(self, fake_capture):
        fake_capture["kwargs"] = {"opened": False}
        with pytest.raises(SourceOpenError, match="Could not open webcam device"):
            WebcamSource(webcam_config()).open()

    def test_transient_read_failures_are_retried(self, fake_capture):
        fake_capture["kwargs"] = {"fail_reads": 2}
        source = WebcamSource(webcam_config(read_retries=5, read_retry_delay_sec=0.0))
        source.open()

        assert source.read() is not None  # recovered after two bad reads
        source.release()

    def test_persistent_read_failure_raises(self, fake_capture):
        fake_capture["kwargs"] = {"fail_reads": 99}
        source = WebcamSource(webcam_config(read_retries=2, read_retry_delay_sec=0.0))
        source.open()

        with pytest.raises(FrameReadError, match="failed to read a frame"):
            source.read()
        source.release()

    def test_context_manager_opens_and_releases(self, fake_capture):
        with WebcamSource(webcam_config()) as source:
            assert source.is_open
            assert source.read() is not None
        assert not source.is_open
        assert FakeCapture.instances[-1].released

    def test_release_is_idempotent(self, fake_capture):
        source = WebcamSource(webcam_config())
        source.open()
        source.release()
        source.release()  # must not raise
        assert not source.is_open

    def test_frames_generator_yields_until_the_stream_breaks(self, fake_capture):
        fake_capture["kwargs"] = {"frames": 3}
        source = WebcamSource(webcam_config(read_retries=0, read_retry_delay_sec=0.0))

        collected = []
        with pytest.raises(FrameReadError):
            for packet in source.frames():
                collected.append(packet)
        assert len(collected) == 3
        source.release()


class TestBackendSelection:
    def test_windows_defaults_to_msmf(self, monkeypatch):
        """Measured: MSMF sustains the camera's rated fps where DSHOW caps out."""
        monkeypatch.setattr("app.sources.webcam.sys.platform", "win32")
        assert WebcamSource(webcam_config())._resolve_backend() is CaptureBackend.MSMF

    def test_linux_defaults_to_v4l2(self, monkeypatch):
        monkeypatch.setattr("app.sources.webcam.sys.platform", "linux")
        assert WebcamSource(webcam_config())._resolve_backend() is CaptureBackend.V4L2

    def test_explicit_backend_wins(self, monkeypatch):
        monkeypatch.setattr("app.sources.webcam.sys.platform", "win32")
        source = WebcamSource(webcam_config(backend="dshow"))
        assert source._resolve_backend() is CaptureBackend.DSHOW


class TestPixelFormat:
    def test_fourcc_is_not_set_by_default(self, fake_capture):
        source = WebcamSource(webcam_config())
        source.open()
        assert cv2.CAP_PROP_FOURCC not in FakeCapture.instances[-1].props
        source.release()

    def test_fourcc_is_applied_when_configured(self, fake_capture):
        source = WebcamSource(webcam_config(fourcc="MJPG"))
        source.open()
        assert FakeCapture.instances[-1].props[cv2.CAP_PROP_FOURCC] > 0
        source.release()

    def test_fourcc_must_be_four_characters(self):
        from app.core.config import ConfigError

        with pytest.raises(ConfigError, match="exactly 4 characters"):
            build_config({"video": {"fourcc": "MJP"}})


def test_source_type_enum_covers_the_documented_set():
    assert {t.value for t in VideoSourceType} == {"webcam", "file", "rtsp"}
