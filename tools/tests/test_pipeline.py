"""End-to-end pipeline behaviour, using a synthetic source and dummy modules.

These tests are the real Phase 1 acceptance check: they prove the core runs
frames through an arbitrary list of modules in dependency order, measures
performance, and survives a misbehaving module — all without a camera, a
display or a model.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.core.frame_processor import FrameProcessor
from computer_vision.core.output import BaseOutputSink, NullSink
from computer_vision.core.pipeline import RunLimits, VisionPipeline
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleStatus, ModuleType
from computer_vision.modules.registry import ModuleRegistry
from computer_vision.sources.base import BaseVideoSource, FramePacket, SourceInfo
from computer_vision.utils.timing import PipelineStats
from tools.tests.conftest import make_module


class SyntheticSource(BaseVideoSource):
    """A camera-free video source producing a fixed number of frames."""

    source_type = "synthetic"

    def __init__(self, frames: int = 5, width: int = 640, height: int = 480) -> None:
        super().__init__("synthetic-0")
        self._total = frames
        self._width = width
        self._height = height
        self.released = False

    def open(self) -> SourceInfo:
        self._is_open = True
        return self.info

    def read(self) -> Optional[FramePacket]:
        if self._frame_id >= self._total:
            self._is_open = False
            return None
        return FramePacket(
            frame=np.zeros((self._height, self._width, 3), dtype=np.uint8),
            frame_id=self._next_frame_id(),
            source_id=self.source_id,
        )

    def release(self) -> None:
        self.released = True
        self._is_open = False

    @property
    def info(self) -> SourceInfo:
        return SourceInfo(
            source_id=self.source_id,
            source_type=self.source_type,
            width=self._width,
            height=self._height,
            fps=30.0,
        )


class RecordingSink(BaseOutputSink):
    """Captures what the pipeline emitted, and can request an early stop."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.contexts: List[FrameContext] = []
        self.closed = False
        self._stop_after = stop_after

    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        self.contexts.append(context)
        if self._stop_after is not None and len(self.contexts) >= self._stop_after:
            return False
        return True

    def close(self) -> None:
        self.closed = True


def make_pipeline(modules=(), frames=5, config_overrides=None, sink=None):
    data = {"processing": {"show_window": False, "target_width": None}}
    if config_overrides:
        data.update(config_overrides)
    config = build_config(data)
    source = SyntheticSource(frames=frames)
    sink = sink or RecordingSink()
    pipeline = VisionPipeline(
        config=config,
        source=source,
        modules=list(modules),
        output=sink,
        processor=FrameProcessor(config.processing),
    )
    return pipeline, source, sink


class TestPipelineCore:
    def test_runs_without_any_modules(self):
        """Phase 1 reality: capture -> context -> output, no AI at all."""
        pipeline, source, sink = make_pipeline(frames=5)
        stats = pipeline.run()

        assert stats.frames_processed == 5
        assert len(sink.contexts) == 5
        assert [c.frame_id for c in sink.contexts] == [1, 2, 3, 4, 5]
        assert source.released and sink.closed

    def test_measures_fps_and_latency(self):
        pipeline, _, _ = make_pipeline(frames=10)
        stats = pipeline.run()

        assert stats.fps > 0
        assert stats.total_latency.avg_ms >= 0
        snapshot = stats.as_dict()
        assert snapshot["frames_processed"] == 10
        assert "latency_ms" in snapshot

    def test_context_carries_source_and_timing_metadata(self):
        pipeline, _, sink = make_pipeline(frames=1)
        pipeline.run()

        context = sink.contexts[0]
        assert context.source_id == "synthetic-0"
        assert context.metadata["original_width"] == 640
        assert "latency_ms" in context.metadata
        assert "fps" in context.metadata

    def test_sink_can_stop_the_run(self):
        pipeline, _, sink = make_pipeline(frames=100, sink=RecordingSink(stop_after=3))
        stats = pipeline.run()

        assert stats.frames_processed == 3

    def test_frame_limit_stops_the_run(self):
        pipeline, _, _ = make_pipeline(frames=100)
        stats = pipeline.run(RunLimits(max_frames=4))

        assert stats.frames_processed == 4

    def test_resize_applies_one_resolution_everywhere(self):
        pipeline, _, sink = make_pipeline(
            frames=1, config_overrides={"processing": {"show_window": False, "target_width": 320}}
        )
        pipeline.run()

        context = sink.contexts[0]
        assert context.shape == (320, 240)  # aspect ratio preserved
        assert context.metadata["original_width"] == 640

    def test_teardown_releases_everything(self):
        module_cls = make_module("person_detection")
        module = module_cls()
        module.initialize()
        module.status = ModuleStatus.READY

        pipeline, source, sink = make_pipeline(modules=[module], frames=2)
        pipeline.run()

        assert module.cleaned_up
        assert source.released
        assert sink.closed


class TestModuleExecution:
    def test_modules_receive_every_frame(self):
        module = make_module("person_detection")()
        module.initialize()
        module.status = ModuleStatus.READY

        pipeline, _, _ = make_pipeline(modules=[module], frames=3)
        pipeline.run()

        assert module.processed_frames == [1, 2, 3]

    def test_modules_run_in_the_given_order(self):
        order: List[str] = []

        class Detector(BaseAIModule):
            module_name = "person_detection"
            module_type = ModuleType.DETECTION
            version = "1.0.0"

            def initialize(self) -> None:
                pass

            def process(self, context: FrameContext) -> ModuleResult:
                order.append("detect")
                context.add_detections(
                    self.module_name, [Detection("person", 0.9, BBox(0, 0, 10, 10))]
                )
                return self.make_result(context)

        class Tracker(BaseAIModule):
            module_name = "person_tracking"
            module_type = ModuleType.TRACKING
            version = "1.0.0"
            requires = ("person_detection",)

            def initialize(self) -> None:
                pass

            def process(self, context: FrameContext) -> ModuleResult:
                order.append("track")
                # The handoff Phase 3 depends on: read the dependency's output.
                assert context.get_detections("person_detection")
                return self.make_result(context)

        ModuleRegistry.register(Detector)
        ModuleRegistry.register(Tracker)
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        "person_tracking": {"enabled": True},
                        "person_detection": {"enabled": True},
                    }
                }
            )
        )

        pipeline, _, _ = make_pipeline(modules=built, frames=2)
        pipeline.run()

        assert order == ["detect", "track", "detect", "track"]

    def test_module_results_land_on_the_context(self):
        module = make_module("person_detection")()
        module.initialize()
        module.status = ModuleStatus.READY

        pipeline, _, sink = make_pipeline(modules=[module], frames=1)
        pipeline.run()

        result = sink.contexts[0].get_result("person_detection")
        assert result.module == "person_detection"
        assert result.ok
        assert result.latency_ms >= 0
        assert result.to_dict()["module"] == "person_detection"

    def test_per_module_latency_is_tracked(self):
        module = make_module("person_detection")()
        module.initialize()
        module.status = ModuleStatus.READY

        pipeline, _, _ = make_pipeline(modules=[module], frames=3)
        stats = pipeline.run()

        assert "person_detection" in stats.module_latency


class TestFailureContainment:
    def test_a_failing_module_does_not_stop_the_pipeline(self):
        """Agent.md §19: one broken optional module must not crash the app."""
        broken = make_module("fire_detection", fail_on_process=True)()
        broken.initialize()
        broken.status = ModuleStatus.READY

        healthy = make_module("person_detection")()
        healthy.initialize()
        healthy.status = ModuleStatus.READY

        pipeline, _, sink = make_pipeline(modules=[broken, healthy], frames=5)
        stats = pipeline.run()

        assert stats.frames_processed == 5
        assert healthy.processed_frames == [1, 2, 3, 4, 5]
        assert "fire_detection" in sink.contexts[0].errors

    def test_a_persistently_failing_module_is_dropped(self):
        broken = make_module("fire_detection", fail_on_process=True)()
        broken.initialize()
        broken.status = ModuleStatus.READY

        pipeline, _, _ = make_pipeline(modules=[broken], frames=15)
        pipeline.run()

        assert broken.status is ModuleStatus.FAILED
        assert broken.consecutive_failures >= 10

    def test_error_is_captured_in_the_result(self, frame_context):
        broken = make_module("fire_detection", fail_on_process=True)()
        result = broken.safe_process(frame_context)

        assert not result.ok
        assert "blew up" in result.error
        assert result.to_dict()["error"]


class TestEndReason:
    """A source that dies mid-run is contained, but its reason must survive.

    Before this, the dashboard only ever said "no frames received" for an RTSP
    camera whose stream failed, hiding what actually went wrong.
    """

    def test_a_source_failure_is_kept_as_the_end_reason(self):
        from computer_vision.sources.base import FrameReadError

        pipeline, source, _ = make_pipeline(frames=5)

        def dead_read():
            raise FrameReadError("RTSP stream rtsp://cam/x?token=*** failed after 5 reconnect attempt(s)")

        source.read = dead_read
        stats = pipeline.run()

        assert stats.frames_processed == 0
        assert "failed after 5 reconnect" in pipeline.end_reason

    def test_a_clean_end_has_no_reason(self):
        pipeline, _, _ = make_pipeline(frames=3)
        pipeline.run()
        assert pipeline.end_reason is None

    def test_a_silent_source_reports_the_stall(self):
        pipeline, source, _ = make_pipeline(
            frames=5, config_overrides={"processing": {"show_window": False, "target_width": None, "stall_timeout_sec": 0.05}}
        )
        source.read = lambda: None
        pipeline.run()
        assert pipeline.end_reason and "no frame arrived" in pipeline.end_reason


class TestNullSink:
    def test_headless_sink_never_stops_the_run(self, frame_context):
        assert NullSink(log_every=1).emit(frame_context, PipelineStats()) is True
