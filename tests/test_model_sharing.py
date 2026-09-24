"""Model sharing: one load, one inference, many modules.

Agent.md §9 forbids duplicating model inference. These tests hold the line:
if sharing regresses, two detectors on the same weights silently double the
per-frame cost again.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import build_config
from app.core.context import FrameContext
from app.models.base import RawDetection
from app.models.factory import (
    create_detector,
    reset_detector_cache,
    shared_detector_count,
)
from app.models.shared import SharedDetector, share_key
from app.modules.detection.object import ObjectDetectionModule
from app.modules.detection.person import PersonDetectionModule
from app.modules.detection.vehicle import VehicleDetectionModule
from app.modules.registry import ModuleRegistry
from tests.conftest import FakeDetectorBackend, inner_backend


def raw(label: str, conf: float = 0.9, box=(10, 10, 60, 200)):
    return RawDetection(label=label, confidence=conf, xyxy=box, class_id=-1)


def frame_ctx(frame=None, frame_id: int = 1) -> FrameContext:
    return FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame,
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=0.0,
    )


def build(cls, **params):
    params.setdefault("backend", "fake")
    module = cls(params)
    module.initialize()
    return module


class TestShareKey:
    def test_same_model_config_shares(self):
        a = {"model": "yolo11n.pt", "device": "cpu", "imgsz": 640}
        b = {"model": "yolo11n.pt", "device": "cpu", "imgsz": 640}
        assert share_key("ultralytics", a) == share_key("ultralytics", b)

    def test_module_level_settings_do_not_prevent_sharing(self):
        """Labels and thresholds are applied after inference, so they are free."""
        a = {"model": "yolo11n.pt", "labels": ["person"], "confidence_threshold": 0.4}
        b = {"model": "yolo11n.pt", "labels": ["car"], "confidence_threshold": 0.8}
        assert share_key("ultralytics", a) == share_key("ultralytics", b)

    @pytest.mark.parametrize(
        "difference",
        [
            {"model": "yolo11m.pt"},
            {"device": "cuda"},
            {"imgsz": 320},
            {"iou": 0.7},
            {"classes": [0]},
        ],
    )
    def test_different_inference_config_does_not_share(self, difference):
        base = {"model": "yolo11n.pt", "device": "cpu", "imgsz": 640}
        other = {**base, **difference}
        assert share_key("ultralytics", base) != share_key("ultralytics", other)

    def test_different_backends_do_not_share(self):
        params = {"model": "yolo11n.pt"}
        assert share_key("ultralytics", params) != share_key("onnx", params)


class TestFactoryCache:
    def test_matching_configs_get_the_same_instance(self):
        first = create_detector("fake", {"model": "a.pt"})
        second = create_detector("fake", {"model": "a.pt"})
        assert first is second
        assert shared_detector_count() == 1

    def test_differing_configs_get_separate_instances(self):
        first = create_detector("fake", {"model": "a.pt"})
        second = create_detector("fake", {"model": "b.pt"})
        assert first is not second
        assert shared_detector_count() == 2

    def test_sharing_can_be_disabled(self):
        first = create_detector("fake", {"model": "a.pt", "share_model": False})
        second = create_detector("fake", {"model": "a.pt", "share_model": False})
        assert first is not second
        assert not isinstance(first, SharedDetector)
        assert shared_detector_count() == 0

    def test_share_model_flag_is_not_forwarded_to_the_backend(self):
        backend = create_detector("fake", {"model": "a.pt", "share_model": False})
        assert "share_model" not in backend.params

    def test_reset_clears_the_cache(self):
        create_detector("fake", {"model": "a.pt"})
        reset_detector_cache()
        assert shared_detector_count() == 0


class TestLoadSharing:
    def test_model_loads_only_once(self):
        FakeDetectorBackend.scripted = []
        person = build(PersonDetectionModule, model="shared.pt")
        vehicle = build(VehicleDetectionModule, model="shared.pt")

        assert person._backend is vehicle._backend
        assert inner_backend(person).load_calls == 1

    def test_different_models_load_separately(self):
        person = build(PersonDetectionModule, model="a.pt")
        vehicle = build(VehicleDetectionModule, model="b.pt")

        assert person._backend is not vehicle._backend
        assert inner_backend(person).load_calls == 1
        assert inner_backend(vehicle).load_calls == 1

    def test_user_count_tracks_modules(self):
        person = build(PersonDetectionModule, model="shared.pt")
        assert person._backend.users == 1

        vehicle = build(VehicleDetectionModule, model="shared.pt")
        assert person._backend.users == 2

        vehicle.cleanup()
        assert person._backend.users == 1

    def test_one_module_cleaning_up_does_not_unload_for_others(self):
        """The bug this guards: first cleanup() pulls the model out from
        under a module that is still running."""
        person = build(PersonDetectionModule, model="shared.pt")
        vehicle = build(VehicleDetectionModule, model="shared.pt")
        backend = inner_backend(person)

        vehicle.cleanup()

        assert backend.is_loaded
        assert backend.unload_calls == 0
        assert person._backend.is_loaded

    def test_last_module_out_unloads(self):
        person = build(PersonDetectionModule, model="shared.pt")
        vehicle = build(VehicleDetectionModule, model="shared.pt")
        backend = inner_backend(person)

        vehicle.cleanup()
        person.cleanup()

        assert backend.unload_calls == 1
        assert not backend.is_loaded


class TestInferenceSharing:
    def test_one_inference_serves_both_modules(self):
        FakeDetectorBackend.scripted = [raw("person"), raw("car")]
        person = build(PersonDetectionModule, model="shared.pt")
        vehicle = build(VehicleDetectionModule, model="shared.pt")
        ctx = frame_ctx()

        person.process(ctx)
        vehicle.process(ctx)

        assert inner_backend(person).frames_seen == 1
        assert person._backend.inference_count == 1
        assert person._backend.cache_hits == 1

    def test_each_module_still_gets_its_own_labels(self):
        """Sharing must not blur the modules' separate outputs."""
        FakeDetectorBackend.scripted = [raw("person"), raw("car"), raw("dog")]
        person = build(PersonDetectionModule, model="shared.pt")
        vehicle = build(VehicleDetectionModule, model="shared.pt")
        objects = build(ObjectDetectionModule, model="shared.pt")
        ctx = frame_ctx()

        for module in (person, vehicle, objects):
            module.process(ctx)

        assert [d.label for d in ctx.get_detections("person_detection")] == ["person"]
        assert [d.label for d in ctx.get_detections("vehicle_detection")] == ["car"]
        assert len(ctx.get_detections("object_detection")) == 3
        assert inner_backend(person).frames_seen == 1

    def test_a_new_frame_triggers_a_new_inference(self):
        FakeDetectorBackend.scripted = [raw("person")]
        person = build(PersonDetectionModule, model="shared.pt")
        vehicle = build(VehicleDetectionModule, model="shared.pt")

        for frame_id in (1, 2, 3):
            ctx = frame_ctx(frame_id=frame_id)
            person.process(ctx)
            vehicle.process(ctx)

        assert inner_backend(person).frames_seen == 3
        assert person._backend.cache_hits == 3

    def test_results_are_not_shared_by_reference(self):
        """One module must not be able to corrupt another's results."""
        FakeDetectorBackend.scripted = [raw("person"), raw("car")]
        backend = create_detector("fake", {"model": "shared.pt"})
        backend.load()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        first = backend.infer(frame)
        first.clear()
        second = backend.infer(frame)

        assert len(second) == 2

    def test_unshared_modules_each_infer(self):
        FakeDetectorBackend.scripted = [raw("person"), raw("car")]
        person = build(PersonDetectionModule, model="shared.pt", share_model=False)
        vehicle = build(VehicleDetectionModule, model="shared.pt", share_model=False)
        ctx = frame_ctx()

        person.process(ctx)
        vehicle.process(ctx)

        assert inner_backend(person).frames_seen == 1
        assert inner_backend(vehicle).frames_seen == 1

    def test_different_models_do_not_share_results(self):
        FakeDetectorBackend.scripted = [raw("person"), raw("car")]
        person = build(PersonDetectionModule, model="a.pt")
        vehicle = build(VehicleDetectionModule, model="b.pt")
        ctx = frame_ctx()

        person.process(ctx)
        vehicle.process(ctx)

        assert inner_backend(person).frames_seen == 1
        assert inner_backend(vehicle).frames_seen == 1


class TestPipelineSharing:
    def test_three_detectors_cost_one_inference_per_frame(self):
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink, SyntheticSource

        FakeDetectorBackend.scripted = [raw("person"), raw("car")]
        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                    "vehicle_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                    "object_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        # Capture before the run: shutdown detaches the backend from modules.
        backend = inner_backend(modules[0])
        shared = modules[0]._backend

        VisionPipeline(
            config=config,
            source=SyntheticSource(frames=5),
            modules=modules,
            output=RecordingSink(),
            processor=FrameProcessor(config.processing),
        ).run()

        # 5 frames x 3 modules would be 15 inferences without sharing.
        assert backend.frames_seen == 5
        assert shared.cache_hits == 10

    def test_shutdown_unloads_exactly_once(self):
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink, SyntheticSource

        FakeDetectorBackend.scripted = [raw("person")]
        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                    "vehicle_detection": {"enabled": True, "backend": "fake", "model": "s.pt"},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        backend = inner_backend(modules[0])

        VisionPipeline(
            config=config,
            source=SyntheticSource(frames=2),
            modules=modules,
            output=RecordingSink(),
            processor=FrameProcessor(config.processing),
        ).run()

        assert backend.unload_calls == 1
        assert not backend.is_loaded
