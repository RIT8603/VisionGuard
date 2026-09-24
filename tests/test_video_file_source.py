"""Video file source.

Tests write real video files with ``cv2.VideoWriter`` and read them back, so
the decode path, EOF handling, looping and seeking are genuinely exercised
rather than mocked.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import CaptureBackend, build_config
from app.sources.base import SourceExhausted, SourceNotOpenError, SourceOpenError
from app.sources.factory import available_source_types, create_video_source
from app.sources.video_file import VideoFileSource

FRAME_SIZE = (320, 240)  # (width, height)


def write_clip(path: Path, frames: int = 12, fps: float = 10.0) -> Path:
    """Write a clip whose Nth frame is filled with brightness N.

    Encoding is lossy, so the marker is read back approximately - which is
    still enough to prove frames arrive in order.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, FRAME_SIZE
    )
    if not writer.isOpened():
        pytest.skip("No MJPG encoder available in this OpenCV build")

    for index in range(frames):
        frame = np.full(
            (FRAME_SIZE[1], FRAME_SIZE[0], 3), (index + 1) * 10, dtype=np.uint8
        )
        writer.write(frame)
    writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("OpenCV produced no video file in this environment")
    return path


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return write_clip(tmp_path / "clip.avi", frames=12, fps=10.0)


def file_config(path: Path, **kwargs):
    return build_config({"video": {"source": "file", "path": str(path), **kwargs}}).video


def open_source(path: Path, **kwargs) -> VideoFileSource:
    source = VideoFileSource(file_config(path, **kwargs))
    source.open()
    return source


class TestFactory:
    def test_file_source_is_now_available(self):
        assert available_source_types()["file"] is True

    def test_factory_builds_a_file_source(self, clip):
        source = create_video_source(file_config(clip))
        assert isinstance(source, VideoFileSource)

    def test_source_id_is_derived_from_the_filename(self, clip):
        assert create_video_source(file_config(clip)).source_id == "file-clip"

    def test_defaults_to_the_ffmpeg_backend(self, clip):
        source = VideoFileSource(file_config(clip))
        assert source._resolve_backend() is CaptureBackend.FFMPEG

    def test_explicit_backend_wins(self, clip):
        source = VideoFileSource(file_config(clip, backend="any"))
        assert source._resolve_backend() is CaptureBackend.ANY


class TestOpen:
    def test_reports_frame_count_fps_and_duration(self, clip):
        source = open_source(clip)

        assert source.frame_count == 12
        assert source.info.fps == pytest.approx(10.0, abs=0.5)
        assert source.duration_sec == pytest.approx(1.2, abs=0.2)
        assert (source.info.width, source.info.height) == FRAME_SIZE
        source.release()

    def test_description_mentions_the_file(self, clip):
        source = open_source(clip)
        assert "clip.avi" in source.info.description
        source.release()

    def test_missing_file_is_an_actionable_error(self, tmp_path):
        source = VideoFileSource(file_config(tmp_path / "nope.avi"))
        with pytest.raises(SourceOpenError, match="Video file not found"):
            source.open()

    def test_directory_instead_of_file_is_rejected(self, tmp_path):
        source = VideoFileSource(file_config(tmp_path))
        with pytest.raises(SourceOpenError, match="is a directory"):
            source.open()

    def test_config_requires_a_path(self):
        from app.core.config import ConfigError

        with pytest.raises(ConfigError, match="path is required"):
            build_config({"video": {"source": "file"}})


class TestReading:
    def test_reads_every_frame_in_order(self, clip):
        source = open_source(clip)

        packets = list(source.frames())

        assert len(packets) == 12
        assert [p.frame_id for p in packets] == list(range(1, 13))
        source.release()

    def test_frames_carry_the_source_id(self, clip):
        source = open_source(clip)
        packet = source.read()
        assert packet.source_id == "file-clip"
        source.release()

    def test_frame_content_advances(self, clip):
        """The encoded brightness marker should increase frame over frame."""
        source = open_source(clip)
        first = source.read().frame.mean()
        for _ in range(5):
            latest = source.read().frame.mean()
        assert latest > first
        source.release()

    def test_end_of_file_raises_source_exhausted(self, clip):
        source = open_source(clip)
        for _ in range(12):
            source.read()

        with pytest.raises(SourceExhausted, match="Reached the end"):
            source.read()
        source.release()

    def test_frames_generator_stops_cleanly_at_eof(self, clip):
        """Unlike a camera, running out is normal - not an error."""
        source = open_source(clip)
        assert len(list(source.frames())) == 12
        source.release()

    def test_read_before_open_is_rejected(self, clip):
        with pytest.raises(SourceNotOpenError):
            VideoFileSource(file_config(clip)).read()

    def test_eof_is_not_retried_with_sleeps(self, clip):
        """A finished file must stop immediately, not stall through retries."""
        import time

        source = open_source(clip, read_retries=5, read_retry_delay_sec=0.5)
        for _ in range(12):
            source.read()

        started = time.perf_counter()
        with pytest.raises(SourceExhausted):
            source.read()
        assert time.perf_counter() - started < 0.2
        source.release()


class TestLooping:
    def test_loop_restarts_instead_of_stopping(self, clip):
        source = open_source(clip, loop=True)

        packets = [source.read() for _ in range(18)]

        assert all(p is not None for p in packets)
        assert [p.frame_id for p in packets] == list(range(1, 19))
        assert source.loops_completed >= 1
        source.release()

    def test_loop_respects_start_frame(self, clip):
        source = open_source(clip, loop=True, start_frame=6)

        for _ in range(20):
            source.read()

        assert source.loops_completed >= 1
        source.release()

    def test_empty_file_does_not_loop_forever(self, tmp_path):
        """A corrupt or empty file must terminate rather than spin."""
        broken = tmp_path / "broken.avi"
        broken.write_bytes(b"not a video")

        source = VideoFileSource(file_config(broken, loop=True))
        with pytest.raises(SourceOpenError):
            source.open()


class TestSeeking:
    def test_start_frame_skips_ahead(self, clip):
        full = open_source(clip)
        total = len(list(full.frames()))
        full.release()

        seeked = open_source(clip, start_frame=8)
        remaining = len(list(seeked.frames()))
        seeked.release()

        assert total == 12
        assert remaining == 4

    def test_seek_moves_the_position(self, clip):
        source = open_source(clip)
        source.read()
        source.seek(9)

        assert len(list(source.frames())) == 3
        source.release()

    def test_seek_past_the_end_is_rejected(self, clip):
        source = open_source(clip)
        with pytest.raises(SourceOpenError, match="Cannot seek"):
            source.seek(999)
        source.release()

    def test_seek_before_open_is_rejected(self, clip):
        with pytest.raises(SourceOpenError, match="before open"):
            VideoFileSource(file_config(clip)).seek(1)

    def test_negative_start_frame_is_rejected(self, clip):
        from app.core.config import ConfigError

        with pytest.raises(ConfigError, match="start_frame"):
            file_config(clip, start_frame=-1)


class TestPacing:
    def test_unthrottled_by_default(self, clip):
        import time

        source = open_source(clip)
        started = time.perf_counter()
        list(source.frames())
        elapsed = time.perf_counter() - started

        # 12 frames at 10 fps would take 1.2s if paced.
        assert elapsed < 0.5
        source.release()

    def test_playback_speed_throttles_to_native_rate(self, clip):
        import time

        source = open_source(clip, playback_speed=1.0)
        started = time.perf_counter()
        for _ in range(5):
            source.read()
        elapsed = time.perf_counter() - started

        # 5 frames at 10 fps is ~0.4s of spacing after the first.
        assert elapsed >= 0.3
        source.release()

    def test_negative_playback_speed_is_rejected(self, clip):
        from app.core.config import ConfigError

        with pytest.raises(ConfigError, match="playback_speed"):
            file_config(clip, playback_speed=-1.0)

    def test_progress_advances(self, clip):
        source = open_source(clip)
        assert source.progress == 0.0
        for _ in range(6):
            source.read()
        assert 0.4 < source.progress < 0.6
        source.release()


class TestPipelineIntegration:
    def test_pipeline_runs_a_file_to_completion(self, clip):
        """The whole point: a repeatable run with a definite end."""
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink

        config = build_config(
            {
                "video": {"source": "file", "path": str(clip)},
                "processing": {"show_window": False, "target_width": None},
            }
        )
        sink = RecordingSink()
        pipeline = VisionPipeline(
            config=config,
            source=create_video_source(config.video),
            modules=[],
            output=sink,
            processor=FrameProcessor(config.processing),
        )

        stats = pipeline.run()

        assert stats.frames_processed == 12
        assert stats.frames_failed == 0
        assert len(sink.contexts) == 12

    def test_two_runs_over_the_same_file_are_identical(self, clip):
        """Reproducibility is the reason this source exists."""
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink

        def run_once():
            config = build_config(
                {
                    "video": {"source": "file", "path": str(clip)},
                    "processing": {"show_window": False, "target_width": None},
                }
            )
            sink = RecordingSink()
            VisionPipeline(
                config=config,
                source=create_video_source(config.video),
                modules=[],
                output=sink,
                processor=FrameProcessor(config.processing),
            ).run()
            return [float(ctx.frame.mean()) for ctx in sink.contexts]

        assert run_once() == run_once()


class TestCliInference:
    def test_video_path_implies_the_file_source(self):
        from app.main import build_parser, infer_source

        args = build_parser().parse_args(["--video-path", "clip.mp4"])
        assert infer_source(args) == "file"

    def test_rtsp_url_implies_the_rtsp_source(self):
        from app.main import build_parser, infer_source

        args = build_parser().parse_args(["--rtsp-url", "rtsp://x"])
        assert infer_source(args) == "rtsp"

    def test_explicit_source_is_not_overridden(self):
        from app.main import build_parser, infer_source

        args = build_parser().parse_args(
            ["--source", "webcam", "--video-path", "clip.mp4"]
        )
        assert infer_source(args) == "webcam"

    def test_no_flags_leaves_the_config_alone(self):
        from app.main import build_parser, infer_source

        assert infer_source(build_parser().parse_args([])) is None
