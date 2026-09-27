"""Person Tracking module and the IoU tracker underneath it.

No model, no camera: the tracker is fed synthetic detections, which is the
only way to assert id persistence deterministically.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.modules.base import ModuleType
from computer_vision.modules.detection.person import PersonDetectionModule
from computer_vision.modules.registry import ModuleRegistry
from computer_vision.modules.tracking.person import PersonTrackingModule
from computer_vision.modules.tracking.tracker_core import IoUTracker, TrackState


def det(x1, y1, x2, y2, conf=0.9, label="person") -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(x1, y1, x2, y2))


def context_with(detections: List[Detection], frame_id: int = 1) -> FrameContext:
    ctx = FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=0.0,
    )
    ctx.add_detections("person_detection", detections)
    return ctx


def build_module(**params) -> PersonTrackingModule:
    module = PersonTrackingModule(params)
    module.initialize()
    return module


# --------------------------------------------------------------------------- #
# The tracker algorithm
# --------------------------------------------------------------------------- #

class TestIoUTrackerIdentity:
    def test_same_object_keeps_its_id(self):
        """The core promise: frame 1 -> id 1, frame 2 -> id 1, frame 3 -> id 1."""
        tracker = IoUTracker(min_hits=1)

        ids = []
        for step in range(3):
            tracks = tracker.update([det(10 + step * 4, 10, 60 + step * 4, 200)])
            ids.append([t.track_id for t in tracks])

        assert ids == [[1], [1], [1]]

    def test_two_objects_get_distinct_ids(self):
        tracker = IoUTracker(min_hits=1)

        tracker.update([det(10, 10, 60, 200), det(300, 10, 350, 200)])
        tracks = tracker.update([det(12, 10, 62, 200), det(302, 10, 352, 200)])

        assert sorted(t.track_id for t in tracks) == [1, 2]

    def test_ids_do_not_swap_when_objects_pass_close(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(100, 10, 150, 200), det(200, 10, 250, 200)])

        # Both move toward each other but keep their own best overlap.
        tracks = tracker.update([det(110, 10, 160, 200), det(190, 10, 240, 200)])
        by_id = {t.track_id: t.bbox.x1 for t in tracks}

        assert by_id[1] == 110  # the left object is still id 1
        assert by_id[2] == 190

    def test_a_new_object_gets_a_new_id(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(10, 10, 60, 200)])
        tracker.update([det(10, 10, 60, 200)])

        tracks = tracker.update([det(10, 10, 60, 200), det(400, 10, 450, 200)])

        assert sorted(t.track_id for t in tracks) == [1, 2]

    def test_ids_are_never_reused(self):
        tracker = IoUTracker(min_hits=1, max_age=0)
        tracker.update([det(10, 10, 60, 200)])
        tracker.update([])          # id 1 dies immediately
        tracks = tracker.update([det(10, 10, 60, 200)])

        assert [t.track_id for t in tracks] == [2]
        assert tracker.total_ids_issued == 2

    def test_far_apart_boxes_are_not_associated(self):
        tracker = IoUTracker(min_hits=1, iou_threshold=0.3)
        tracker.update([det(10, 10, 60, 200)])

        tracks = tracker.update([det(500, 300, 550, 470)])

        assert [t.track_id for t in tracks] == [2]


class TestIoUTrackerLifecycle:
    def test_min_hits_delays_reporting(self):
        tracker = IoUTracker(min_hits=3)

        assert tracker.update([det(10, 10, 60, 200)]) == []
        assert tracker.update([det(10, 10, 60, 200)]) == []
        assert len(tracker.update([det(10, 10, 60, 200)])) == 1

    def test_one_frame_false_positive_is_never_reported(self):
        tracker = IoUTracker(min_hits=2, max_age=0)

        assert tracker.update([det(10, 10, 60, 200)]) == []
        assert tracker.update([]) == []
        assert tracker.active_tracks == []

    def test_track_survives_a_brief_gap_and_keeps_its_id(self):
        tracker = IoUTracker(min_hits=1, max_age=5)
        tracker.update([det(10, 10, 60, 200)])
        tracker.update([det(14, 10, 64, 200)])

        assert tracker.update([]) == []          # occluded, not reported
        assert tracker.update([]) == []

        recovered = tracker.update([det(26, 10, 76, 200)])
        assert [t.track_id for t in recovered] == [1]

    def test_track_is_dropped_after_max_age(self):
        tracker = IoUTracker(min_hits=1, max_age=2)
        tracker.update([det(10, 10, 60, 200)])

        for _ in range(4):
            tracker.update([])

        assert tracker.active_tracks == []
        assert [t.track_id for t in tracker.update([det(10, 10, 60, 200)])] == [2]

    def test_coasting_tracks_are_hidden_by_default(self):
        tracker = IoUTracker(min_hits=1, max_age=5)
        tracker.update([det(10, 10, 60, 200)])

        assert tracker.update([]) == []
        assert len(tracker.active_tracks) == 1   # alive, just not reported

    def test_coasting_tracks_can_be_published(self):
        tracker = IoUTracker(min_hits=1, max_age=5, publish_coasting=True)
        tracker.update([det(10, 10, 60, 200)])

        coasting = tracker.update([])

        assert len(coasting) == 1
        assert coasting[0].time_since_update == 1

    def test_hits_and_age_are_counted(self):
        tracker = IoUTracker(min_hits=1)
        for _ in range(3):
            tracks = tracker.update([det(10, 10, 60, 200)])

        assert tracks[0].hits == 3
        assert tracks[0].age == 2

    def test_reset_clears_everything(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(10, 10, 60, 200)])
        tracker.reset()

        assert tracker.active_tracks == []
        assert [t.track_id for t in tracker.update([det(10, 10, 60, 200)])] == [1]


class TestIoUTrackerEdgeCases:
    def test_empty_first_frame_is_fine(self):
        assert IoUTracker().update([]) == []

    def test_labels_are_never_crossed(self):
        """A car must not inherit a person's id even at perfect overlap."""
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(10, 10, 60, 200, label="person")])

        tracks = tracker.update([det(10, 10, 60, 200, label="car")])

        assert len(tracks) == 1
        assert tracks[0].label == "car"
        assert tracks[0].track_id == 2

    def test_output_is_ordered_by_track_id(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(10, 10, 60, 200), det(300, 10, 350, 200)])
        tracks = tracker.update([det(300, 10, 350, 200), det(10, 10, 60, 200)])

        assert [t.track_id for t in tracks] == [1, 2]

    def test_velocity_prediction_helps_fast_motion(self):
        """A consistently moving box is re-acquired after a missed frame."""
        tracker = IoUTracker(min_hits=1, max_age=3, iou_threshold=0.3)
        for step in range(4):
            tracker.update([det(10 + step * 20, 10, 60 + step * 20, 200)])

        tracker.update([])                                   # one dropped frame
        recovered = tracker.update([det(110, 10, 160, 200)])  # continued motion

        assert [t.track_id for t in recovered] == [1]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"iou_threshold": 0.0},
            {"iou_threshold": 1.5},
            {"max_age": -1},
            {"min_hits": 0},
        ],
    )
    def test_invalid_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            IoUTracker(**kwargs)


class TestTrackState:
    def test_converts_to_the_shared_track_type(self):
        state = TrackState(track_id=7, label="person", bbox=BBox(1, 2, 3, 4), confidence=0.9)
        track = state.as_track("person_tracking")

        assert track.track_id == 7
        assert track.source_module == "person_tracking"
        assert track.to_dict()["class"] == "person"


# --------------------------------------------------------------------------- #
# The module
# --------------------------------------------------------------------------- #

class TestModuleIdentity:
    def test_declares_its_dependency_on_detection(self):
        assert PersonTrackingModule.module_name == "person_tracking"
        assert PersonTrackingModule.module_type is ModuleType.TRACKING
        assert PersonTrackingModule.requires == ("person_detection",)

    def test_registry_resolves_it_by_name(self):
        assert ModuleRegistry.resolve("person_tracking") is PersonTrackingModule

    def test_initialize_loads_no_model(self):
        """Tracking must stay free of detection concerns - no weights at all."""
        module = build_module()
        assert module._tracker is not None
        assert not hasattr(module, "_backend")

    def test_cleanup_releases_the_tracker(self):
        module = build_module()
        module.cleanup()
        assert module._tracker is None
        assert module.active_track_ids == []


class TestModuleProcess:
    def test_consumes_detections_and_publishes_tracks(self):
        module = build_module(min_hits=1)
        ctx = context_with([det(10, 10, 60, 200)])

        result = module.process(ctx)

        tracks = ctx.get_tracks("person_tracking")
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].source_module == "person_tracking"
        assert result.data["count"] == 1

    def test_track_schema_matches_the_spec(self):
        module = build_module(min_hits=1)
        ctx = context_with([det(10, 20, 60, 200, conf=0.91)])

        payload = module.process(ctx).to_dict()

        track = payload["tracks"][0]
        assert track["track_id"] == 1
        assert track["class"] == "person"
        assert track["bbox"] == [10.0, 20.0, 60.0, 200.0]
        assert track["confidence"] == 0.91

    def test_ids_persist_across_frames_through_the_module(self):
        module = build_module(min_hits=1)

        seen = []
        for frame_id in range(1, 4):
            ctx = context_with([det(10 + frame_id * 3, 10, 60 + frame_id * 3, 200)], frame_id)
            module.process(ctx)
            seen.append([t.track_id for t in ctx.get_tracks("person_tracking")])

        assert seen == [[1], [1], [1]]

    def test_no_detections_yields_no_tracks(self):
        module = build_module(min_hits=1)
        ctx = context_with([])

        result = module.process(ctx)

        assert result.ok
        assert result.data["count"] == 0
        assert ctx.get_tracks("person_tracking") == []

    def test_ignores_detections_from_other_modules(self):
        """It consumes person_detection only, not whatever else is on the context."""
        module = build_module(min_hits=1)
        ctx = context_with([])
        ctx.add_detections("fire_detection", [det(10, 10, 60, 200, label="fire")])

        module.process(ctx)

        assert ctx.get_tracks("person_tracking") == []

    def test_reports_the_detector_it_consumed(self):
        """Lets the output layer avoid drawing the same box twice."""
        module = build_module(min_hits=1)
        ctx = context_with([det(10, 10, 60, 200)])

        result = module.process(ctx)

        assert result.data["consumed_from"] == "person_detection"

    def test_params_reach_the_tracker(self):
        module = build_module(iou_threshold=0.55, max_age=7, min_hits=4)

        assert module._tracker.iou_threshold == 0.55
        assert module._tracker.max_age == 7
        assert module._tracker.min_hits == 4


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

class TestRegistryWiring:
    def test_detection_is_ordered_before_tracking(self):
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        # Listed backwards on purpose.
                        "person_tracking": {"enabled": True},
                        "person_detection": {"enabled": True, "backend": "fake"},
                    }
                }
            )
        )
        assert [m.module_name for m in built] == ["person_detection", "person_tracking"]

    def test_tracking_without_detection_is_rejected(self):
        from computer_vision.modules.registry import ModuleDependencyError

        with pytest.raises(ModuleDependencyError, match="person_detection"):
            ModuleRegistry.build_enabled(
                build_config({"modules": {"person_tracking": {"enabled": True}}})
            )

    def test_detection_alone_is_fine(self):
        built = ModuleRegistry.build_enabled(
            build_config(
                {"modules": {"person_detection": {"enabled": True, "backend": "fake"}}}
            )
        )
        assert [m.module_name for m in built] == ["person_detection"]


class TestFullPipeline:
    """Detection -> context -> tracking, driven by the real pipeline."""

    def _run(self, sequence, frames, **tracking_params):
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import VisionPipeline
        from computer_vision.inference.base import RawDetection
        from tests.conftest import FakeDetectorBackend
        from tests.test_pipeline import RecordingSink, SyntheticSource

        FakeDetectorBackend.scripted_sequence = [
            [
                RawDetection(label="person", confidence=0.9, xyxy=box, class_id=0)
                for box in boxes
            ]
            for boxes in sequence
        ]

        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake"},
                    "person_tracking": {"enabled": True, **tracking_params},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        sink = RecordingSink()
        pipeline = VisionPipeline(
            config=config,
            source=SyntheticSource(frames=frames),
            modules=modules,
            output=sink,
            processor=FrameProcessor(config.processing),
        )
        pipeline.run()
        return sink.contexts, pipeline

    def test_a_person_keeps_one_id_across_the_whole_run(self):
        """Frame 1 -> Track 1, Frame 2 -> Track 1, Frame 3 -> Track 1."""
        walking = [[(10 + step * 6, 10, 60 + step * 6, 200)] for step in range(6)]

        contexts, _ = self._run(walking, frames=6, min_hits=1)

        ids_per_frame = [
            [t.track_id for t in ctx.get_tracks("person_tracking")] for ctx in contexts
        ]
        assert ids_per_frame == [[1]] * 6

    def test_two_people_keep_separate_ids(self):
        both = [
            [(10 + step * 4, 10, 60 + step * 4, 200), (300, 10, 350, 200)]
            for step in range(5)
        ]

        contexts, _ = self._run(both, frames=5, min_hits=1)

        assert [
            sorted(t.track_id for t in ctx.get_tracks("person_tracking"))
            for ctx in contexts
        ] == [[1, 2]] * 5

    def test_modules_run_detection_then_tracking(self):
        contexts, _ = self._run([[(10, 10, 60, 200)]], frames=2, min_hits=1)

        ctx = contexts[-1]
        # Both modules wrote to the same context, each under its own name.
        assert ctx.get_detections("person_detection")
        assert ctx.get_tracks("person_tracking")
        assert ctx.get_result("person_detection").ok
        assert ctx.get_result("person_tracking").ok

    def test_both_module_latencies_are_measured(self):
        _, pipeline = self._run([[(10, 10, 60, 200)]], frames=3, min_hits=1)

        assert "person_detection" in pipeline.stats.module_latency
        assert "person_tracking" in pipeline.stats.module_latency

    def test_context_summary_carries_tracks(self):
        import json

        contexts, _ = self._run([[(10, 10, 60, 200)]], frames=2, min_hits=1)
        summary = contexts[-1].summary()

        json.dumps(summary)
        assert summary["tracks"]["person_tracking"][0]["track_id"] == 1

    def test_a_person_leaving_frame_ends_their_track(self):
        sequence = [[(10, 10, 60, 200)]] * 3 + [[]] * 4

        contexts, _ = self._run(sequence, frames=7, min_hits=1, max_age=1)

        counts = [len(ctx.get_tracks("person_tracking")) for ctx in contexts]
        assert counts[:3] == [1, 1, 1]
        assert counts[-1] == 0
