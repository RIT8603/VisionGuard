"""Person Detection module and the detector-backend abstraction.

The module is tested against a fake backend, so these tests need no model, no
torch and no network. The Ultralytics adapter's own translation logic is tested
separately against a fake Results object.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import FrameContext
from computer_vision.inference.base import (
    BackendError,
    DetectorBackend,
    ModelLoadError,
    RawDetection,
)
from computer_vision.inference.factory import available_backends, create_detector
from computer_vision.modules.base import ModuleInitializationError, ModuleType
from computer_vision.modules.detection.person import COCO_PERSON_CLASS_ID, PersonDetectionModule
from computer_vision.modules.registry import ModuleRegistry
from tests.conftest import FakeDetectorBackend, inner_backend


#: The shared fake backend lives in conftest so tracking tests can use it too.
FakeBackend = FakeDetectorBackend


def person(conf: float, box=(10, 10, 60, 200)) -> RawDetection:
    return RawDetection(
        label="person", confidence=conf, xyxy=box, class_id=COCO_PERSON_CLASS_ID
    )


def build_module(**params) -> PersonDetectionModule:
    params.setdefault("backend", "fake")
    module = PersonDetectionModule(params)
    module.initialize()
    return module


class TestBackendFactory:
    def test_ultralytics_is_registered(self):
        assert "ultralytics" in available_backends()

    def test_unknown_backend_is_reported(self):
        with pytest.raises(BackendError, match="Unknown detector backend"):
            create_detector("magic-vision")

    def test_backend_is_built_but_not_loaded(self):
        backend = create_detector("fake", {})
        assert isinstance(getattr(backend, "inner", backend), FakeBackend)
        assert backend.is_loaded is False


class TestModuleIdentity:
    def test_declares_the_interface_contract(self):
        assert PersonDetectionModule.module_name == "person_detection"
        assert PersonDetectionModule.module_type is ModuleType.DETECTION
        assert PersonDetectionModule.requires == ()

    def test_registry_resolves_it_by_name(self):
        assert ModuleRegistry.resolve("person_detection") is PersonDetectionModule

    def test_registry_builds_it_from_config(self):
        built = ModuleRegistry.build_enabled(
            build_config(
                {"modules": {"person_detection": {"enabled": True, "backend": "fake"}}}
            )
        )
        assert len(built) == 1
        assert isinstance(built[0], PersonDetectionModule)


class TestInitialize:
    def test_loads_the_backend(self):
        module = build_module()
        assert module._backend.is_loaded
        assert inner_backend(module).load_calls == 1

    def test_selects_people_by_label_not_by_class_id(self):
        """Measured: restricting the model to class ids saves nothing (+0.2%),
        and filtering by label here lets modules share one inference."""
        module = build_module()
        assert module._backend.params.get("classes") is None
        assert module._labels == ("person",)

    def test_explicit_classes_are_passed_through(self):
        module = build_module(classes=[0, 1])
        assert module._backend.params["classes"] == [0, 1]

    def test_backend_load_failure_becomes_a_module_error(self):
        FakeBackend.fail_on_load = True
        with pytest.raises(ModuleInitializationError, match="no weights here"):
            build_module()

    def test_unknown_backend_becomes_a_module_error(self):
        with pytest.raises(ModuleInitializationError, match="Unknown detector backend"):
            PersonDetectionModule({"backend": "nope"}).initialize()

    def test_cleanup_unloads_the_model(self):
        module = build_module()
        backend = inner_backend(module)
        module.cleanup()
        assert backend.unload_calls == 1
        assert module._backend is None


class TestProcess:
    def test_publishes_detections_to_the_context(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.94), person(0.81, (100, 20, 150, 210))]
        module = build_module()

        result = module.process(frame_context)

        published = frame_context.get_detections("person_detection")
        assert len(published) == 2
        assert all(det.source_module == "person_detection" for det in published)
        assert result.data["count"] == 2

    def test_result_matches_the_standard_schema(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.94)]
        module = build_module()

        payload = module.process(frame_context).to_dict()

        assert payload["module"] == "person_detection"
        assert payload["module_type"] == "detection"
        assert payload["detections"][0]["class"] == "person"
        assert payload["detections"][0]["confidence"] == 0.94
        assert len(payload["detections"][0]["bbox"]) == 4

    def test_low_confidence_is_dropped(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.94), person(0.20)]
        module = build_module(confidence_threshold=0.5)

        module.process(frame_context)

        assert len(frame_context.get_detections("person_detection")) == 1

    def test_other_classes_are_dropped(self, frame_context: FrameContext):
        """Defence in depth: the module filters even if the backend did not."""
        FakeBackend.scripted = [
            person(0.9),
            RawDetection(label="car", confidence=0.99, xyxy=(0, 0, 50, 50), class_id=2),
        ]
        module = build_module()

        module.process(frame_context)

        labels = [d.label for d in frame_context.get_detections("person_detection")]
        assert labels == ["person"]

    def test_boxes_are_clipped_to_the_frame(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.9, (-50, -20, 9999, 9999))]
        module = build_module()

        module.process(frame_context)

        bbox = frame_context.get_detections("person_detection")[0].bbox
        assert bbox.as_tuple() == (0.0, 0.0, 640.0, 480.0)

    def test_degenerate_boxes_are_dropped(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.9, (700, 500, 800, 600))]  # fully outside
        module = build_module()

        module.process(frame_context)

        assert frame_context.get_detections("person_detection") == []

    def test_min_box_area_is_applied(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.9, (0, 0, 10, 10)), person(0.9, (0, 0, 100, 200))]
        module = build_module(min_box_area=1000)

        module.process(frame_context)

        assert len(frame_context.get_detections("person_detection")) == 1

    def test_detections_are_sorted_by_confidence(self, frame_context: FrameContext):
        FakeBackend.scripted = [person(0.51), person(0.94), person(0.72)]
        module = build_module()

        module.process(frame_context)

        confidences = [d.confidence for d in frame_context.get_detections("person_detection")]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_frame_produces_an_empty_result(self, frame_context: FrameContext):
        FakeBackend.scripted = []
        module = build_module()

        result = module.process(frame_context)

        assert result.ok
        assert result.data["count"] == 0
        assert frame_context.get_detections("person_detection") == []

    def test_backend_failure_is_contained_by_safe_process(self, frame_context, monkeypatch):
        module = build_module()
        monkeypatch.setattr(
            module._backend, "infer", lambda frame: (_ for _ in ()).throw(RuntimeError("cuda oom"))
        )

        result = module.safe_process(frame_context)

        assert not result.ok
        assert "cuda oom" in result.error
        assert "person_detection" in frame_context.errors


class TestUltralyticsAdapter:
    """The translation layer, without importing torch or ultralytics."""

    def _backend(self):
        from computer_vision.inference.ultralytics_backend import UltralyticsDetector

        backend = UltralyticsDetector({"model": "yolo11n.pt"})
        backend._names = {0: "person", 2: "car"}
        return backend

    def test_converts_boxes_confidences_and_classes(self):
        class FakeTensor:
            def __init__(self, array):
                self._array = np.asarray(array)

            def cpu(self):
                return self

            def numpy(self):
                return self._array

        class FakeBoxes:
            def __init__(self):
                self.xyxy = FakeTensor([[10.0, 20.0, 60.0, 200.0]])
                self.conf = FakeTensor([0.9412])
                self.cls = FakeTensor([0.0])

            def __len__(self):
                return 1

        class FakeResult:
            boxes = FakeBoxes()

        detections = self._backend()._to_raw_detections(FakeResult())

        assert len(detections) == 1
        assert detections[0].label == "person"
        assert detections[0].class_id == 0
        assert detections[0].confidence == pytest.approx(0.9412, abs=1e-4)
        assert detections[0].xyxy == (10.0, 20.0, 60.0, 200.0)

    def test_no_boxes_yields_no_detections(self):
        class Empty:
            boxes = None

        assert self._backend()._to_raw_detections(Empty()) == []

    def test_infer_before_load_is_rejected(self):
        with pytest.raises(ModelLoadError, match="before load"):
            self._backend().infer(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_description_names_model_and_device(self):
        assert "yolo11n.pt" in self._backend().description
