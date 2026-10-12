"""Phase 8 tracking modules: object and vehicle tracking.

Fed scripted detections, which is the only way to assert id persistence
deterministically. No model, no camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference.base import RawDetection
from computer_vision.modules.base import ModuleType
from computer_vision.modules.registry import ModuleDependencyError, ModuleRegistry
from computer_vision.modules.tracking.base import BaseTrackingModule
from computer_vision.modules.tracking.object import ObjectTrackingModule
from computer_vision.modules.tracking.person import PersonTrackingModule
from computer_vision.modules.tracking.vehicle import VehicleTrackingModule
from tools.tests.conftest import FakeDetectorBackend

TRACKING_MODULES = [PersonTrackingModule, ObjectTrackingModule, VehicleTrackingModule]


def det(label: str, box, conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(*box))


def ctx_with(source: str, detections, frame_id: int = 1) -> FrameContext:
    ctx = FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=0.0,
    )
    ctx.add_detections(source, detections)
    return ctx


def build(cls, **params):
    module = cls(params)
    module.initialize()
    return module


# --------------------------------------------------------------------------- #
# Contract shared by every tracker
# --------------------------------------------------------------------------- #

class TestTrackingContract:
    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_is_a_tracking_module(self, cls):
        assert issubclass(cls, BaseTrackingModule)
        assert cls.module_type is ModuleType.TRACKING

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_requires_matches_its_source(self, cls):
        """A mismatch would let the registry order or validate the wrong module."""
        assert cls.requires == (cls.source_module,)

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_source_is_a_detection_module(self, cls):
        source_cls = ModuleRegistry.resolve(cls.source_module)
        assert source_cls.module_type is ModuleType.DETECTION

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_loads_no_model(self, cls):
        module = build(cls)
        assert module._tracker is not None
        assert not hasattr(module, "_backend")

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_cleanup_releases_the_tracker(self, cls):
        module = build(cls)
        module.cleanup()
        assert module._tracker is None
        assert module.active_track_ids == []

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_reports_the_detector_it_consumed(self, cls):
        module = build(cls, min_hits=1)
        ctx = ctx_with(cls.source_module, [])
        assert module.process(ctx).data["consumed_from"] == cls.source_module

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_params_reach_the_tracker(self, cls):
        module = build(cls, iou_threshold=0.55, max_age=7, min_hits=4)
        assert module._tracker.iou_threshold == 0.55
        assert module._tracker.max_age == 7
        assert module._tracker.min_hits == 4

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_ids_persist_across_frames(self, cls):
        label = "car" if cls is VehicleTrackingModule else "person"
        module = build(cls, min_hits=1)

        seen = []
        for frame_id in range(1, 4):
            ctx = ctx_with(
                cls.source_module,
                [det(label, (10 + frame_id * 4, 10, 60 + frame_id * 4, 200))],
                frame_id,
            )
            module.process(ctx)
            seen.append([t.track_id for t in ctx.get_tracks(cls.module_name)])

        assert seen == [[1], [1], [1]]

    @pytest.mark.parametrize("cls", TRACKING_MODULES)
    def test_ignores_other_modules_detections(self, cls):
        module = build(cls, min_hits=1)
        ctx = ctx_with(cls.source_module, [])
        ctx.add_detections("something_else", [det("person", (10, 10, 60, 200))])

        module.process(ctx)

        assert ctx.get_tracks(cls.module_name) == []


# --------------------------------------------------------------------------- #
# Object tracking
# --------------------------------------------------------------------------- #

class TestObjectTracking:
    def test_tracks_several_classes_at_once(self):
        module = build(ObjectTrackingModule, min_hits=1)
        ctx = ctx_with(
            "object_detection",
            [det("person", (10, 10, 60, 200)), det("car", (300, 100, 500, 300))],
        )

        module.process(ctx)

        tracks = {t.label: t.track_id for t in ctx.get_tracks("object_tracking")}
        assert set(tracks) == {"person", "car"}
        assert tracks["person"] != tracks["car"]

    def test_ids_are_not_reassigned_across_classes(self):
        """A car must not inherit a person's id even at perfect overlap."""
        module = build(ObjectTrackingModule, min_hits=1)
        box = (10, 10, 60, 200)

        module.process(ctx_with("object_detection", [det("person", box)], 1))
        ctx = ctx_with("object_detection", [det("car", box)], 2)
        module.process(ctx)

        tracks = ctx.get_tracks("object_tracking")
        assert len(tracks) == 1
        assert tracks[0].label == "car"
        assert tracks[0].track_id == 2

    def test_summary_breaks_down_by_class(self):
        module = build(ObjectTrackingModule, min_hits=1)
        ctx = ctx_with(
            "object_detection",
            [
                det("person", (10, 10, 60, 200)),
                det("person", (100, 10, 150, 200)),
                det("car", (300, 100, 500, 300)),
            ],
        )

        data = module.process(ctx).data

        assert data["count"] == 3
        assert data["by_class"] == {"person": 2, "car": 1}

    def test_classes_keep_independent_continuity(self):
        module = build(ObjectTrackingModule, min_hits=1)

        for frame_id in range(1, 4):
            ctx = ctx_with(
                "object_detection",
                [
                    det("person", (10 + frame_id * 3, 10, 60 + frame_id * 3, 200)),
                    det("car", (300 + frame_id * 8, 100, 500 + frame_id * 8, 300)),
                ],
                frame_id,
            )
            module.process(ctx)

        final = {t.label: t.track_id for t in ctx.get_tracks("object_tracking")}
        assert final == {"person": 1, "car": 2}


# --------------------------------------------------------------------------- #
# Vehicle tracking
# --------------------------------------------------------------------------- #

class TestVehicleTracking:
    def test_tracks_vehicles(self):
        module = build(VehicleTrackingModule, min_hits=1)

        for frame_id in range(1, 4):
            ctx = ctx_with(
                "vehicle_detection",
                [det("car", (100 + frame_id * 10, 100, 300 + frame_id * 10, 250))],
                frame_id,
            )
            module.process(ctx)

        assert [t.track_id for t in ctx.get_tracks("vehicle_tracking")] == [1]

    def test_default_settings_suit_faster_motion(self):
        """Config defaults live in config.yaml, but the module must accept them."""
        module = build(VehicleTrackingModule, iou_threshold=0.25, max_age=45)
        assert module._tracker.iou_threshold == 0.25
        assert module._tracker.max_age == 45

    def test_lower_iou_threshold_survives_a_fast_vehicle(self):
        """A vehicle moving most of its own width per frame still keeps its id."""
        strict = build(VehicleTrackingModule, min_hits=1, iou_threshold=0.6)
        lenient = build(VehicleTrackingModule, min_hits=1, iou_threshold=0.2)

        boxes = [(0 + i * 90, 100, 200 + i * 90, 250) for i in range(3)]
        for module in (strict, lenient):
            for frame_id, box in enumerate(boxes, start=1):
                ctx = ctx_with("vehicle_detection", [det("car", box)], frame_id)
                module.process(ctx)
            module._final = [t.track_id for t in ctx.get_tracks(module.module_name)]

        assert lenient._final == [1]      # kept its identity
        assert strict._final != [1]       # lost it and started a new track


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #

class TestRegistryWiring:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("object_tracking", ObjectTrackingModule),
            ("vehicle_tracking", VehicleTrackingModule),
        ],
    )
    def test_resolves_by_name(self, name, cls):
        assert ModuleRegistry.resolve(name) is cls

    @pytest.mark.parametrize(
        "tracker,detector",
        [
            ("object_tracking", "object_detection"),
            ("vehicle_tracking", "vehicle_detection"),
        ],
    )
    def test_detector_runs_first(self, tracker, detector):
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        tracker: {"enabled": True},          # listed first
                        detector: {"enabled": True, "backend": "fake"},
                    }
                }
            )
        )
        assert [m.module_name for m in built] == [detector, tracker]

    @pytest.mark.parametrize(
        "tracker,detector",
        [
            ("object_tracking", "object_detection"),
            ("vehicle_tracking", "vehicle_detection"),
        ],
    )
    def test_tracking_without_its_detector_is_rejected(self, tracker, detector):
        with pytest.raises(ModuleDependencyError, match=detector):
            ModuleRegistry.build_enabled(
                build_config({"modules": {tracker: {"enabled": True}}})
            )

    def test_person_tracking_is_not_satisfied_by_object_detection(self):
        """Each tracker names its own detector; a different one will not do."""
        with pytest.raises(ModuleDependencyError, match="person_detection"):
            ModuleRegistry.build_enabled(
                build_config(
                    {
                        "modules": {
                            "person_tracking": {"enabled": True},
                            "object_detection": {"enabled": True, "backend": "fake"},
                        }
                    }
                )
            )


# --------------------------------------------------------------------------- #
# Everything together
# --------------------------------------------------------------------------- #

class TestFullPipeline:
    def test_two_trackers_keep_separate_id_spaces(self):
        """person #1 and vehicle #1 are different things, and that is fine."""
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import VisionPipeline
        from tools.tests.test_pipeline import RecordingSink, SyntheticSource

        FakeDetectorBackend.scripted = [
            RawDetection(label="person", confidence=0.9, xyxy=(10, 10, 60, 200), class_id=0),
            RawDetection(label="car", confidence=0.85, xyxy=(300, 100, 500, 300), class_id=2),
        ]
        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                    "vehicle_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                    "person_tracking": {"enabled": True, "min_hits": 1},
                    "vehicle_tracking": {"enabled": True, "min_hits": 1},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        sink = RecordingSink()
        VisionPipeline(
            config=config,
            source=SyntheticSource(frames=4),
            modules=modules,
            output=sink,
            processor=FrameProcessor(config.processing),
        ).run()

        ctx = sink.contexts[-1]
        people = ctx.get_tracks("person_tracking")
        vehicles = ctx.get_tracks("vehicle_tracking")

        assert [t.track_id for t in people] == [1]
        assert [t.track_id for t in vehicles] == [1]
        assert people[0].label == "person"
        assert vehicles[0].label == "car"
        # Same id number, different namespaces - no collision.
        assert people[0].source_module != vehicles[0].source_module

    def test_detectors_and_trackers_order_correctly(self):
        config = build_config(
            {
                "modules": {
                    # Scrambled on purpose.
                    "vehicle_tracking": {"enabled": True},
                    "person_tracking": {"enabled": True},
                    "object_tracking": {"enabled": True},
                    "object_detection": {"enabled": True, "backend": "fake"},
                    "vehicle_detection": {"enabled": True, "backend": "fake"},
                    "person_detection": {"enabled": True, "backend": "fake"},
                }
            }
        )
        order = [m.module_name for m in ModuleRegistry.build_enabled(config)]

        for tracker, detector in (
            ("person_tracking", "person_detection"),
            ("object_tracking", "object_detection"),
            ("vehicle_tracking", "vehicle_detection"),
        ):
            assert order.index(detector) < order.index(tracker)

    def test_tracked_boxes_are_not_drawn_twice(self):
        """Each tracker marks its detector consumed, so no duplicate boxes."""
        from computer_vision.core.output import untracked_detections

        FakeDetectorBackend.scripted = [
            RawDetection(label="car", confidence=0.9, xyxy=(300, 100, 500, 300), class_id=2),
        ]
        config = build_config(
            {
                "modules": {
                    "vehicle_detection": {"enabled": True, "backend": "fake"},
                    "vehicle_tracking": {"enabled": True, "min_hits": 1},
                }
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        ctx = FrameContext(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=1,
            source_id="test",
            monotonic_ts=0.0,
        )
        for module in modules:
            module.safe_process(ctx)

        assert len(ctx.get_tracks("vehicle_tracking")) == 1
        assert untracked_detections(ctx) == []
