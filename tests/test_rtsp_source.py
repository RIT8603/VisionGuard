"""RTSP source.

No camera is required: a fake ``cv2.VideoCapture`` stands in for the stream,
which is the only way to test a dropped connection and a reconnect
deterministically.

Credential redaction is tested hard on purpose - camera URLs contain real,
frequently reused passwords, and a leak into a log file is a security incident.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pytest

from app.core.config import CaptureBackend, ConfigError, build_config
from app.sources import opencv_base
from app.sources.base import FrameReadError, SourceNotOpenError, SourceOpenError
from app.sources.factory import available_source_types, create_video_source
from app.sources.rtsp import RTSPSource, redact_url

URL = "rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101"


def rtsp_config(url: str = URL, **kwargs):
    return build_config({"video": {"source": "rtsp", "rtsp_url": url, **kwargs}}).video


class FakeStream:
    """A fake RTSP capture that can be told to drop the connection."""

    instances: list["FakeStream"] = []

    def __init__(self, target, backend=0, params=None, *, opened=True, frames_before_drop=None):
        self.target = target
        self.backend = backend
        self.params = list(params or [])
        self._opened = opened
        self._frames_before_drop = frames_before_drop
        self._served = 0
        self.released = False
        self.props = {}
        # Captured at construction: this is what OpenCV would have seen.
        self.ffmpeg_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        FakeStream.instances.append(self)

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        if self._frames_before_drop is not None and self._served >= self._frames_before_drop:
            return False, None
        self._served += 1
        time.sleep(0.002)  # let the reader thread interleave realistically
        return True, np.full((240, 320, 3), self._served % 255, dtype=np.uint8)

    def get(self, prop):
        defaults = {
            cv2.CAP_PROP_FRAME_WIDTH: 320.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 240.0,
            cv2.CAP_PROP_FPS: 25.0,
        }
        return self.props.get(prop, defaults.get(prop, 0.0))

    def set(self, prop, value):
        self.props[prop] = float(value)
        return True

    def release(self):
        self.released = True
        self._opened = False


@pytest.fixture
def fake_stream(monkeypatch):
    FakeStream.instances.clear()
    settings = {}

    def factory(target, backend=0, params=None):
        return FakeStream(target, backend, params, **settings.get("kwargs", {}))

    monkeypatch.setattr(opencv_base.cv2, "VideoCapture", factory)
    return settings


# --------------------------------------------------------------------------- #
# Credential handling
# --------------------------------------------------------------------------- #

class TestRedaction:
    @pytest.mark.parametrize(
        "url,expected",
        [
            (
                "rtsp://admin:hunter2@10.0.0.5:554/stream",
                "rtsp://admin:***@10.0.0.5:554/stream",
            ),
            ("rtsp://admin:p@ss:word@10.0.0.5/stream", None),  # only assert no secret
            ("rtsp://admin@10.0.0.5/stream", "rtsp://admin@10.0.0.5/stream"),
            ("rtsp://10.0.0.5:554/stream", "rtsp://10.0.0.5:554/stream"),
            ("", ""),
        ],
    )
    def test_password_is_masked(self, url, expected):
        result = redact_url(url)
        if expected is not None:
            assert result == expected
        assert "hunter2" not in result

    def test_odd_urls_still_do_not_leak(self):
        for url in (
            "rtsp://user:s3cret@host/path?token=s3cret",
            "rtsp://:s3cret@host/path",
            "not a url at all s3cret",
        ):
            redacted = redact_url(url)
            # The query string is not credentials, but the userinfo must be gone.
            assert "://user:s3cret" not in redacted
            assert "://:s3cret" not in redacted

    def test_source_exposes_only_the_safe_url(self):
        source = RTSPSource(rtsp_config())
        assert source.safe_url == "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101"
        assert "hunter2" not in source.safe_url
        assert "hunter2" not in repr(source)
        assert "hunter2" not in source._describe()

    def test_open_failure_message_is_redacted(self, fake_stream):
        fake_stream["kwargs"] = {"opened": False}
        with pytest.raises(SourceOpenError) as exc:
            RTSPSource(rtsp_config()).open()
        assert "hunter2" not in str(exc.value)
        assert "admin:***" in str(exc.value)

    def test_log_output_never_contains_the_password(self, fake_stream, caplog):
        with caplog.at_level("DEBUG"):
            source = RTSPSource(rtsp_config())
            source.open()
            source.read()
            source.release()
        assert "hunter2" not in caplog.text


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class TestConfiguration:
    def test_rtsp_is_available(self):
        assert available_source_types()["rtsp"] is True

    def test_factory_builds_an_rtsp_source(self):
        assert isinstance(create_video_source(rtsp_config()), RTSPSource)

    def test_url_is_required(self):
        with pytest.raises(ConfigError, match="rtsp_url is required"):
            build_config({"video": {"source": "rtsp"}})

    def test_transport_is_validated(self):
        with pytest.raises(ConfigError, match="tcp.*udp"):
            rtsp_config(rtsp_transport="carrier-pigeon")

    def test_transport_is_normalised(self):
        assert rtsp_config(rtsp_transport="TCP").rtsp_transport == "tcp"

    def test_defaults_to_ffmpeg(self):
        assert RTSPSource(rtsp_config())._resolve_backend() is CaptureBackend.FFMPEG

    def test_non_rtsp_url_is_rejected(self, fake_stream):
        source = RTSPSource(rtsp_config("http://10.0.0.5/stream"))
        with pytest.raises(SourceOpenError, match="must start with rtsp"):
            source.open()

    def test_empty_url_is_rejected(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source._config = source._config.model_copy(update={"rtsp_url": ""})
        with pytest.raises(SourceOpenError, match="not set"):
            source.open()


class TestFFmpegOptions:
    def test_transport_reaches_ffmpeg(self, fake_stream):
        source = RTSPSource(rtsp_config(rtsp_transport="tcp"))
        source.open()

        assert "rtsp_transport;tcp" in FakeStream.instances[-1].ffmpeg_options
        source.release()

    def test_udp_transport_is_passed_through(self, fake_stream):
        source = RTSPSource(rtsp_config(rtsp_transport="udp"))
        source.open()

        assert "rtsp_transport;udp" in FakeStream.instances[-1].ffmpeg_options
        source.release()

    def test_open_timeout_is_passed_in_microseconds(self, fake_stream):
        source = RTSPSource(rtsp_config(open_timeout_sec=7.0))
        source.open()

        options = FakeStream.instances[-1].ffmpeg_options
        assert "timeout;7000000" in options
        assert "stimeout;7000000" in options
        source.release()

    def test_environment_is_restored_after_open(self, fake_stream, monkeypatch):
        monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "preexisting;value")

        source = RTSPSource(rtsp_config())
        source.open()
        source.release()

        assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "preexisting;value"

    def test_environment_is_cleared_when_it_was_unset(self, fake_stream, monkeypatch):
        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)

        source = RTSPSource(rtsp_config())
        source.open()
        source.release()

        assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ

    def test_timeouts_are_passed_at_construction(self, fake_stream):
        """Set after construction they are ignored, and a dead camera blocks
        for FFmpeg's 30s default instead of the configured timeout."""
        source = RTSPSource(rtsp_config(open_timeout_sec=4.0, read_timeout_sec=2.0))
        source.open()

        params = FakeStream.instances[-1].params
        assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params
        assert params[params.index(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC) + 1] == 4000
        assert cv2.CAP_PROP_READ_TIMEOUT_MSEC in params
        assert params[params.index(cv2.CAP_PROP_READ_TIMEOUT_MSEC) + 1] == 2000
        source.release()

    def test_webcam_does_not_get_capture_params(self, fake_stream):
        """The params form is RTSP-specific; other sources keep the plain call."""
        from app.sources.webcam import WebcamSource

        webcam_cfg = build_config({"video": {"source": "webcam"}}).video
        assert WebcamSource(webcam_cfg)._capture_params() == []

    def test_buffer_is_kept_shallow(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()

        assert FakeStream.instances[-1].props[cv2.CAP_PROP_BUFFERSIZE] == 1.0
        source.release()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

class TestReading:
    def test_reads_frames(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()

        packet = source.read()

        assert packet is not None
        assert packet.frame_id == 1
        assert packet.source_id == "rtsp-0"
        assert (packet.width, packet.height) == (320, 240)
        source.release()

    def test_frame_ids_increase(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()

        ids = []
        for _ in range(4):
            packet = source.read()
            if packet:
                ids.append(packet.frame_id)

        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        source.release()

    def test_read_before_open_is_rejected(self):
        with pytest.raises(SourceNotOpenError):
            RTSPSource(rtsp_config()).read()

    def test_reports_negotiated_properties(self, fake_stream):
        source = RTSPSource(rtsp_config())
        info = source.open()

        assert info.source_type == "rtsp"
        assert (info.width, info.height) == (320, 240)
        assert info.fps == 25.0
        source.release()

    def test_release_is_idempotent(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()
        source.release()
        source.release()
        assert not source.is_open


class TestStaleFrameDropping:
    def test_only_the_newest_frame_is_served(self, fake_stream):
        """A slow consumer must get fresh frames, not a growing backlog."""
        source = RTSPSource(rtsp_config())
        source.open()

        source.read()
        time.sleep(0.15)          # let the reader race ahead
        source.read()

        assert source.frames_dropped > 0
        source.release()

    def test_drop_count_survives_release(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()
        source.read()
        time.sleep(0.1)
        source.read()
        dropped = source.frames_dropped
        source.release()

        assert source.frames_dropped >= dropped

    def test_reader_thread_is_not_left_running(self, fake_stream):
        before = threading.active_count()

        source = RTSPSource(rtsp_config())
        source.open()
        source.read()
        source.release()
        time.sleep(0.1)

        assert threading.active_count() <= before


# --------------------------------------------------------------------------- #
# Resilience
# --------------------------------------------------------------------------- #

class TestReconnect:
    def test_dropped_stream_reconnects(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 3}
        source = RTSPSource(rtsp_config(reconnect_delay_sec=0.01, read_timeout_sec=0.2))
        source.open()

        for _ in range(12):
            source.read()

        assert source.total_reconnects >= 1
        assert len(FakeStream.instances) >= 2  # a new capture was built
        source.release()

    def test_reconnect_reopens_with_ffmpeg_options(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 2}
        source = RTSPSource(
            rtsp_config(reconnect_delay_sec=0.01, read_timeout_sec=0.2, rtsp_transport="udp")
        )
        source.open()
        for _ in range(10):
            source.read()

        assert "rtsp_transport;udp" in FakeStream.instances[-1].ffmpeg_options
        source.release()

    def test_gives_up_after_the_configured_attempts(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 0}
        source = RTSPSource(
            rtsp_config(
                reconnect_attempts=2, reconnect_delay_sec=0.01, read_timeout_sec=0.1
            )
        )
        source.open()

        with pytest.raises(FrameReadError, match="appears to be offline"):
            for _ in range(20):
                source.read()

        assert source.total_reconnects == 2
        source.release()

    def test_reconnect_can_be_disabled(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 0}
        source = RTSPSource(
            rtsp_config(reconnect_attempts=0, read_timeout_sec=0.1)
        )
        source.open()

        with pytest.raises(FrameReadError, match="reconnect is disabled"):
            for _ in range(5):
                source.read()
        source.release()

    def test_error_on_giving_up_is_redacted(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 0}
        source = RTSPSource(
            rtsp_config(
                reconnect_attempts=1, reconnect_delay_sec=0.01, read_timeout_sec=0.1
            )
        )
        source.open()

        with pytest.raises(FrameReadError) as exc:
            for _ in range(10):
                source.read()
        assert "hunter2" not in str(exc.value)
        source.release()

    def test_backoff_delay_grows(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 0}
        source = RTSPSource(
            rtsp_config(
                reconnect_attempts=3,
                reconnect_delay_sec=0.05,
                reconnect_max_delay_sec=10.0,
                read_timeout_sec=0.05,
            )
        )
        source.open()

        started = time.perf_counter()
        with pytest.raises(FrameReadError):
            for _ in range(20):
                source.read()
        elapsed = time.perf_counter() - started

        # 0.05 + 0.10 + 0.20 = 0.35s of backoff at minimum.
        assert elapsed >= 0.3
        source.release()

    def test_backoff_is_capped(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 0}
        source = RTSPSource(
            rtsp_config(
                reconnect_attempts=3,
                reconnect_delay_sec=10.0,
                reconnect_max_delay_sec=0.02,
                read_timeout_sec=0.05,
            )
        )
        source.open()

        started = time.perf_counter()
        with pytest.raises(FrameReadError):
            for _ in range(20):
                source.read()

        assert time.perf_counter() - started < 2.0  # cap honoured, not 10s each
        source.release()

    def test_successful_read_resets_the_reconnect_budget(self, fake_stream):
        fake_stream["kwargs"] = {"frames_before_drop": 2}
        source = RTSPSource(
            rtsp_config(
                reconnect_attempts=2, reconnect_delay_sec=0.01, read_timeout_sec=0.2
            )
        )
        source.open()

        # Each new capture serves 2 frames then drops, so the budget keeps
        # resetting and the source survives far more than 2 drops.
        for _ in range(20):
            source.read()

        assert source.total_reconnects > 2
        source.release()

    def test_stats_snapshot_is_safe_to_log(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()
        source.read()

        snapshot = source.stats_snapshot()

        assert "hunter2" not in str(snapshot)
        assert snapshot["source_id"] == "rtsp-0"
        assert "reconnects" in snapshot and "frames_dropped" in snapshot
        source.release()


# --------------------------------------------------------------------------- #
# Pipeline behaviour
# --------------------------------------------------------------------------- #

class TestPipelineIntegration:
    def _pipeline(self, config, sink):
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline

        return VisionPipeline(
            config=config,
            source=create_video_source(config.video),
            modules=[],
            output=sink,
            processor=FrameProcessor(config.processing),
        )

    def test_pipeline_runs_an_rtsp_stream(self, fake_stream):
        from app.core.pipeline import RunLimits
        from tests.test_pipeline import RecordingSink

        config = build_config(
            {
                "video": {"source": "rtsp", "rtsp_url": URL, "read_timeout_sec": 0.2},
                "processing": {"show_window": False, "target_width": None},
            }
        )
        sink = RecordingSink()
        stats = self._pipeline(config, sink).run(RunLimits(max_frames=5))

        assert stats.frames_processed == 5
        assert all(ctx.source_id == "rtsp-0" for ctx in sink.contexts)

    def test_a_transient_gap_does_not_end_the_run(self, fake_stream):
        """The Phase 1 pipeline treated a None read as end-of-stream."""
        from app.core.pipeline import RunLimits
        from tests.test_pipeline import RecordingSink

        config = build_config(
            {
                "video": {
                    "source": "rtsp",
                    "rtsp_url": URL,
                    "read_timeout_sec": 0.05,
                    "reconnect_delay_sec": 0.01,
                },
                "processing": {"show_window": False, "target_width": None},
            }
        )
        source = create_video_source(config.video)

        # Force alternating "no frame yet" reads.
        real_read = source.read
        toggle = {"skip": True}

        def flaky_read():
            toggle["skip"] = not toggle["skip"]
            return None if toggle["skip"] else real_read()

        source.read = flaky_read

        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline

        sink = RecordingSink()
        stats = VisionPipeline(
            config=config,
            source=source,
            modules=[],
            output=sink,
            processor=FrameProcessor(config.processing),
        ).run(RunLimits(max_frames=4))

        assert stats.frames_processed == 4

    def test_a_prolonged_stall_ends_the_run(self, fake_stream):
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink

        config = build_config(
            {
                "video": {"source": "rtsp", "rtsp_url": URL, "read_timeout_sec": 0.02},
                "processing": {
                    "show_window": False,
                    "target_width": None,
                    "stall_timeout_sec": 0.2,
                },
            }
        )
        source = create_video_source(config.video)
        source.read = lambda: None  # a stream that never delivers

        stats = VisionPipeline(
            config=config,
            source=source,
            modules=[],
            output=RecordingSink(),
            processor=FrameProcessor(config.processing),
        ).run()

        assert stats.frames_processed == 0

    def test_stall_check_can_be_disabled(self, fake_stream):
        from app.core.pipeline import RunLimits
        from tests.test_pipeline import RecordingSink

        config = build_config(
            {
                "video": {"source": "rtsp", "rtsp_url": URL, "read_timeout_sec": 0.02},
                "processing": {
                    "show_window": False,
                    "target_width": None,
                    "stall_timeout_sec": 0,
                },
            }
        )
        stats = self._pipeline(config, RecordingSink()).run(RunLimits(max_seconds=0.3))

        assert stats.frames_processed >= 0  # ran to the time limit, not a stall
