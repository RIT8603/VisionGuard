"""Phase 6 detection modules: object, vehicle, crowd, PPE, fire, smoke.

All tested against the shared fake backend, so no model or network is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference.base import RawDetection
from computer_vision.modules.base import ModuleInitializationError, ModuleType
from computer_vision.modules.detection.base import COCO_CLASS_IDS, LabelFilteredDetectionModule
from computer_vision.modules.detection.crowd import CrowdDetectionModule
from computer_vision.modules.detection.fire import FireDetectionModule
from computer_vision.modules.detection.object import ObjectDetectionModule
from computer_vision.modules.detection.person import PersonDetectionModule
from computer_vision.modules.detection.ppe import PPEDetectionModule
from computer_vision.modules.detection.smoke import SmokeDetectionModule
from computer_vision.modules.detection.vehicle import VehicleDetectionModule
from computer_vision.modules.registry import ModuleDependencyError, ModuleRegistry
from tools.tests.conftest import FakeDetectorBackend, inner_backend

STOCK_MODULES = [ObjectDetectionModule, VehicleDetectionModule, PersonDetectionModule]
CUSTOM_MODEL_MODULES = [PPEDetectionModule, FireDetectionModule, SmokeDetectionModule]


def raw(label: str, conf: float = 0.9, box=(10, 10, 60, 200), class_id: int = -1):
    return RawDetection(label=label, confidence=conf, xyxy=box, class_id=class_id)


def build(module_cls, **params):
    params.setdefault("backend", "fake")
    module = module_cls(params)
    module.initialize()
    return module


def frame_ctx(frame_id: int = 1) -> FrameContext:
    return FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=0.0,
    )


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #

class TestRegistry:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("object_detection", ObjectDetectionModule),
            ("vehicle_detection", VehicleDetectionModule),
            ("crowd_detection", CrowdDetectionModule),
            ("ppe_detection", PPEDetectionModule),
            ("fire_detection", FireDetectionModule),
            ("smoke_detection", SmokeDetectionModule),
        ],
    )
    def test_module_resolves_by_name(self, name, cls):
        assert ModuleRegistry.resolve(name) is cls

    def test_every_catalogue_entry_is_loadable(self):
        """A name in BUILTIN_MODULES that cannot be imported is a broken build."""
        for name in ModuleRegistry.known_names():
            assert ModuleRegistry.resolve(name) is not None

    def test_names_are_unique(self):
        names = [
            cls.module_name
            for cls in STOCK_MODULES + CUSTOM_MODEL_MODULES + [CrowdDetectionModule]
        ]
        assert len(names) == len(set(names))

    def test_detectors_declare_no_dependencies(self):
        for cls in STOCK_MODULES + CUSTOM_MODEL_MODULES:
            assert cls.requires == ()
            assert cls.module_type is ModuleType.DETECTION


# --------------------------------------------------------------------------- #
# Object detection
# --------------------------------------------------------------------------- #

class TestObjectDetection:
    def test_reports_every_class_by_default(self):
        FakeDetectorBackend.scripted = [raw("person"), raw("car"), raw("dog")]
        module = build(ObjectDetectionModule)
        ctx = frame_ctx()

        module.process(ctx)

        labels = {d.label for d in ctx.get_detections("object_detection")}
        assert labels == {"person", "car", "dog"}

    def test_labels_can_narrow_the_output(self):
        FakeDetectorBackend.scripted = [raw("person"), raw("car"), raw("dog")]
        module = build(ObjectDetectionModule, labels=["car", "dog"])
        ctx = frame_ctx()

        module.process(ctx)

        assert {d.label for d in ctx.get_detections("object_detection")} == {"car", "dog"}

    def test_summary_counts_by_class(self):
        FakeDetectorBackend.scripted = [raw("car"), raw("car"), raw("person")]
        module = build(ObjectDetectionModule)

        result = module.process(frame_ctx())

        assert result.data["count"] == 3
        assert result.data["by_class"] == {"car": 2, "person": 1}

    def test_no_class_restriction_is_sent_to_the_model(self):
        """Selection is by label, so one inference can serve several modules."""
        module = build(ObjectDetectionModule)
        assert module._backend.params.get("classes") is None


# --------------------------------------------------------------------------- #
# Vehicle detection
# --------------------------------------------------------------------------- #

class TestVehicleDetection:
    def test_reports_vehicles_only(self):
        FakeDetectorBackend.scripted = [raw("car"), raw("bus"), raw("person"), raw("dog")]
        module = build(VehicleDetectionModule)
        ctx = frame_ctx()

        module.process(ctx)

        assert {d.label for d in ctx.get_detections("vehicle_detection")} == {"car", "bus"}

    def test_filters_by_label_not_by_model_class_id(self):
        """Label filtering keeps the shared inference usable by every module."""
        module = build(VehicleDetectionModule)
        assert module._backend.params.get("classes") is None

    def test_custom_model_labels_are_honoured(self):
        """A custom model has its own class names; labels still select them."""
        FakeDetectorBackend.scripted = [raw("forklift"), raw("car")]
        module = build(
            VehicleDetectionModule, model="computer_vision/Model/custom/forklifts.pt", labels=["forklift"]
        )
        ctx = frame_ctx()

        module.process(ctx)

        assert [d.label for d in ctx.get_detections("vehicle_detection")] == ["forklift"]

    def test_labels_outside_coco_still_work(self):
        FakeDetectorBackend.scripted = [raw("car"), raw("forklift")]
        module = build(VehicleDetectionModule, labels=["car", "forklift"])
        ctx = frame_ctx()

        module.process(ctx)

        assert len(ctx.get_detections("vehicle_detection")) == 2

    def test_does_not_collide_with_person_detection(self):
        """Two detectors on one frame keep separate namespaces."""
        FakeDetectorBackend.scripted = [raw("car"), raw("person")]
        ctx = frame_ctx()

        build(VehicleDetectionModule).process(ctx)
        build(PersonDetectionModule).process(ctx)

        assert [d.label for d in ctx.get_detections("vehicle_detection")] == ["car"]
        assert [d.label for d in ctx.get_detections("person_detection")] == ["person"]
        assert len(ctx.get_detections()) == 2


# --------------------------------------------------------------------------- #
# Modules needing custom weights
# --------------------------------------------------------------------------- #

class TestCustomModelModules:
    @pytest.mark.parametrize("cls", CUSTOM_MODEL_MODULES)
    def test_enabling_without_a_model_fails_with_instructions(self, cls):
        with pytest.raises(ModuleInitializationError) as exc:
            cls({"backend": "fake"}).initialize()

        message = str(exc.value)
        assert cls.module_name in message
        assert "model:" in message
        assert "custom-trained" in message

    @pytest.mark.parametrize("cls", CUSTOM_MODEL_MODULES)
    def test_works_once_a_model_is_supplied(self, cls):
        module = build(cls, model=f"computer_vision/Model/custom/{cls.module_name}.pt")
        assert module._backend.is_loaded

    def test_ppe_reports_configured_items(self):
        FakeDetectorBackend.scripted = [raw("helmet"), raw("vest"), raw("person")]
        module = build(PPEDetectionModule, model="computer_vision/Model/custom/ppe.pt")
        ctx = frame_ctx()

        module.process(ctx)

        assert {d.label for d in ctx.get_detections("ppe_detection")} == {"helmet", "vest"}

    def test_ppe_label_names_can_be_overridden(self):
        """Datasets disagree on naming: hardhat vs helmet."""
        FakeDetectorBackend.scripted = [raw("hardhat"), raw("safety-vest")]
        module = build(
            PPEDetectionModule, model="computer_vision/Model/custom/ppe.pt", labels=["hardhat", "safety-vest"]
        )
        ctx = frame_ctx()

        module.process(ctx)

        assert len(ctx.get_detections("ppe_detection")) == 2

    def test_fire_and_smoke_stay_separate(self):
        FakeDetectorBackend.scripted = [raw("fire"), raw("smoke")]
        ctx = frame_ctx()

        build(FireDetectionModule, model="computer_vision/Model/custom/fire.pt").process(ctx)
        build(SmokeDetectionModule, model="computer_vision/Model/custom/smoke.pt").process(ctx)

        assert [d.label for d in ctx.get_detections("fire_detection")] == ["fire"]
        assert [d.label for d in ctx.get_detections("smoke_detection")] == ["smoke"]

    def test_custom_model_path_reaches_the_backend(self):
        module = build(FireDetectionModule, model="computer_vision/Model/custom/custom_fire.pt")
        assert module._backend.params["model"] == "computer_vision/Model/custom/custom_fire.pt"


# --------------------------------------------------------------------------- #
# Crowd detection
# --------------------------------------------------------------------------- #

def people(ctx: FrameContext, boxes, conf=0.9) -> None:
    ctx.add_detections(
        "person_detection",
        [Detection(label="person", confidence=conf, bbox=BBox(*b)) for b in boxes],
    )


class TestCrowdDetection:
    def test_loads_no_model(self):
        module = CrowdDetectionModule({})
        module.initialize()
        assert not hasattr(module, "_backend")

    def test_declares_its_dependency(self):
        assert CrowdDetectionModule.requires == ("person_detection",)
        assert CrowdDetectionModule.module_type is ModuleType.ANALYSIS

    def test_counts_people_from_the_detector(self):
        module = CrowdDetectionModule({"threshold": 3})
        module.initialize()
        ctx = frame_ctx()
        people(ctx, [(0, 0, 50, 100), (60, 0, 110, 100)])

        result = module.process(ctx)

        assert result.data["count"] == 2
        assert result.data["is_crowded"] is False

    def test_flags_a_crowd_at_the_threshold(self):
        module = CrowdDetectionModule({"threshold": 3})
        module.initialize()
        ctx = frame_ctx()
        people(ctx, [(i * 20, 0, i * 20 + 15, 100) for i in range(3)])

        assert module.process(ctx).data["is_crowded"] is True

    def test_empty_frame_is_not_a_crowd(self):
        module = CrowdDetectionModule({"threshold": 1})
        module.initialize()

        result = module.process(frame_ctx())

        assert result.data["count"] == 0
        assert result.data["is_crowded"] is False
        assert result.data["occupancy"] == 0.0
        assert result.data["mean_confidence"] == 0.0

    def test_occupancy_reflects_coverage(self):
        module = CrowdDetectionModule({})
        module.initialize()
        ctx = frame_ctx()  # 640x480 = 307200 px
        people(ctx, [(0, 0, 320, 480)])  # exactly half the frame

        assert module.process(ctx).data["occupancy"] == pytest.approx(0.5, abs=0.01)

    def test_occupancy_is_capped_at_one(self):
        module = CrowdDetectionModule({})
        module.initialize()
        ctx = frame_ctx()
        people(ctx, [(0, 0, 640, 480)] * 3)  # heavy overlap

        assert module.process(ctx).data["occupancy"] == 1.0

    def test_peak_count_persists_across_frames(self):
        module = CrowdDetectionModule({})
        module.initialize()

        busy = frame_ctx(1)
        people(busy, [(i * 10, 0, i * 10 + 8, 50) for i in range(6)])
        module.process(busy)

        quiet = frame_ctx(2)
        people(quiet, [(0, 0, 10, 50)])
        result = module.process(quiet)

        assert result.data["count"] == 1
        assert result.data["peak_count"] == 6

    def test_ignores_other_modules_detections(self):
        module = CrowdDetectionModule({})
        module.initialize()
        ctx = frame_ctx()
        ctx.add_detections(
            "vehicle_detection",
            [Detection(label="car", confidence=0.9, bbox=BBox(0, 0, 50, 50))],
        )

        assert module.process(ctx).data["count"] == 0

    def test_does_not_republish_detections(self):
        """Counting must not duplicate boxes on the frame."""
        module = CrowdDetectionModule({})
        module.initialize()
        ctx = frame_ctx()
        people(ctx, [(0, 0, 50, 100)])

        module.process(ctx)

        assert len(ctx.get_detections()) == 1
        assert ctx.get_detections("crowd_detection") == []

    def test_invalid_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            CrowdDetectionModule({"threshold": 0}).initialize()

    def test_requires_person_detection_to_be_enabled(self):
        with pytest.raises(ModuleDependencyError, match="person_detection"):
            ModuleRegistry.build_enabled(
                build_config({"modules": {"crowd_detection": {"enabled": True}}})
            )

    def test_runs_after_person_detection(self):
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        "crowd_detection": {"enabled": True},
                        "person_detection": {"enabled": True, "backend": "fake"},
                    }
                }
            )
        )
        assert [m.module_name for m in built] == ["person_detection", "crowd_detection"]


# --------------------------------------------------------------------------- #
# Shared base behaviour
# --------------------------------------------------------------------------- #

class TestSharedBase:
    def test_all_detectors_share_the_base(self):
        for cls in STOCK_MODULES + CUSTOM_MODEL_MODULES:
            assert issubclass(cls, LabelFilteredDetectionModule)

    @pytest.mark.parametrize("cls", STOCK_MODULES)
    def test_confidence_threshold_applies(self, cls):
        label = cls.default_labels[0] if cls.default_labels else "car"
        FakeDetectorBackend.scripted = [raw(label, 0.95), raw(label, 0.30)]
        module = build(cls, confidence_threshold=0.5)
        ctx = frame_ctx()

        module.process(ctx)

        assert len(ctx.get_detections(cls.module_name)) == 1

    @pytest.mark.parametrize("cls", STOCK_MODULES)
    def test_boxes_are_clipped(self, cls):
        label = cls.default_labels[0] if cls.default_labels else "car"
        FakeDetectorBackend.scripted = [raw(label, 0.9, (-50, -50, 5000, 5000))]
        module = build(cls)
        ctx = frame_ctx()

        module.process(ctx)

        bbox = ctx.get_detections(cls.module_name)[0].bbox
        assert bbox.as_tuple() == (0.0, 0.0, 640.0, 480.0)

    @pytest.mark.parametrize("cls", STOCK_MODULES)
    def test_results_sort_by_confidence(self, cls):
        label = cls.default_labels[0] if cls.default_labels else "car"
        FakeDetectorBackend.scripted = [raw(label, 0.4), raw(label, 0.95), raw(label, 0.7)]
        module = build(cls)
        ctx = frame_ctx()

        module.process(ctx)

        confs = [d.confidence for d in ctx.get_detections(cls.module_name)]
        assert confs == sorted(confs, reverse=True)

    @pytest.mark.parametrize("cls", STOCK_MODULES)
    def test_cleanup_unloads_the_model(self, cls):
        module = build(cls)
        backend = inner_backend(module)
        module.cleanup()
        assert backend.unload_calls == 1
        assert module._backend is None

    @pytest.mark.parametrize("cls", STOCK_MODULES)
    def test_failure_is_contained(self, cls, monkeypatch):
        module = build(cls)
        monkeypatch.setattr(
            module._backend,
            "infer",
            lambda frame: (_ for _ in ()).throw(RuntimeError("model exploded")),
        )
        ctx = frame_ctx()

        result = module.safe_process(ctx)

        assert not result.ok
        assert "model exploded" in result.error
        assert cls.module_name in ctx.errors


# --------------------------------------------------------------------------- #
# Multiple detectors together
# --------------------------------------------------------------------------- #

class TestCombinedPipeline:
    def test_four_modules_cooperate_on_one_frame(self):
        """person + vehicle + crowd + tracking, each in its own namespace."""
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import VisionPipeline
        from tools.tests.test_pipeline import RecordingSink, SyntheticSource

        FakeDetectorBackend.scripted = [
            raw("person", 0.9, (10, 10, 60, 200)),
            raw("person", 0.8, (100, 10, 150, 200)),
            raw("car", 0.85, (300, 100, 500, 300)),
        ]
        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake"},
                    "vehicle_detection": {"enabled": True, "backend": "fake"},
                    "crowd_detection": {"enabled": True, "threshold": 2},
                    "person_tracking": {"enabled": True, "min_hits": 1},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        sink = RecordingSink()
        VisionPipeline(
            config=config,
            source=SyntheticSource(frames=3),
            modules=modules,
            output=sink,
            processor=FrameProcessor(config.processing),
        ).run()

        ctx = sink.contexts[-1]
        assert len(ctx.get_detections("person_detection")) == 2
        assert len(ctx.get_detections("vehicle_detection")) == 1
        assert len(ctx.get_tracks("person_tracking")) == 2
        assert ctx.get_result("crowd_detection").data["is_crowded"] is True

    def test_dependencies_order_everything_correctly(self):
        config = build_config(
            {
                "modules": {
                    # Deliberately scrambled.
                    "crowd_detection": {"enabled": True},
                    "person_tracking": {"enabled": True},
                    "vehicle_detection": {"enabled": True, "backend": "fake"},
                    "person_detection": {"enabled": True, "backend": "fake"},
                }
            }
        )
        order = [m.module_name for m in ModuleRegistry.build_enabled(config)]

        assert order.index("person_detection") < order.index("person_tracking")
        assert order.index("person_detection") < order.index("crowd_detection")

    def test_disabled_custom_modules_are_never_constructed(self):
        """A disabled fire_detection must not complain about its missing model."""
        config = build_config(
            {
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake"},
                    "fire_detection": {"enabled": False},
                    "ppe_detection": {"enabled": False},
                }
            }
        )
        built = ModuleRegistry.build_enabled(config)
        assert [m.module_name for m in built] == ["person_detection"]
