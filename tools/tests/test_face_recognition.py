"""Phase 7: face detection, recognition, the identity store, unknown person.

Run against fake backends with deterministic embeddings, so no face model and
no real face data are needed.

The identity store's privacy properties are asserted, not just described:
embeddings-only, no pickle on load, deletable, and no implicit enrolment.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import BBox, Detection, FrameContext, Severity
from computer_vision.identity.store import IdentityStore, IdentityStoreError
from computer_vision.inference import face_factory
from computer_vision.inference.face_base import (
    BackendError,
    FaceDetectorBackend,
    FaceEmbedderBackend,
    RawFace,
    cosine_similarity,
)
from computer_vision.modules.base import ModuleInitializationError, ModuleType
from computer_vision.modules.recognition.face_detection import FaceDetectionModule
from computer_vision.modules.recognition.face_recognition import FaceRecognitionModule
from computer_vision.modules.recognition.unknown_person import (
    EVENT_UNKNOWN_PERSON,
    UnknownPersonModule,
)
from computer_vision.modules.registry import ModuleDependencyError, ModuleRegistry

DIM = 8


def vec(seed: int, dim: int = DIM) -> np.ndarray:
    """A deterministic unit-ish embedding."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


class FakeFaceDetector(FaceDetectorBackend):
    name = "fake"
    scripted: list = []
    fail_on_load = False

    def load(self) -> None:
        if FakeFaceDetector.fail_on_load:
            raise BackendError("no face model")
        self._loaded = True

    def detect(self, frame):
        return list(FakeFaceDetector.scripted)


class FakeFaceEmbedder(FaceEmbedderBackend):
    name = "fake"
    default_threshold = 0.5
    #: Maps a face's x1 coordinate to the embedding returned for it, so tests
    #: can place "person A" and "person B" in known positions.
    by_x: dict = {}

    def load(self) -> None:
        self._loaded = True

    def embed(self, frame, face: RawFace):
        return FakeFaceEmbedder.by_x.get(int(face.xyxy[0]), vec(999))


@pytest.fixture(autouse=True)
def fake_face_backends(monkeypatch):
    FakeFaceDetector.scripted = []
    FakeFaceDetector.fail_on_load = False
    FakeFaceEmbedder.by_x = {}
    monkeypatch.setitem(
        face_factory.FACE_DETECTORS, "fake", "tools.tests.test_face_recognition:FakeFaceDetector"
    )
    monkeypatch.setitem(
        face_factory.FACE_EMBEDDERS, "fake", "tools.tests.test_face_recognition:FakeFaceEmbedder"
    )
    yield


def a_face(x1=100, conf=0.9) -> RawFace:
    return RawFace(
        confidence=conf,
        xyxy=(x1, 50, x1 + 60, 120),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        raw_row=np.zeros(15, dtype=np.float32),
    )


def frame_ctx(frame_id: int = 1, ts: float = 0.0) -> FrameContext:
    return FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=ts,
    )


# --------------------------------------------------------------------------- #
# Identity store
# --------------------------------------------------------------------------- #

class TestIdentityStore:
    def test_starts_empty(self):
        store = IdentityStore()
        assert len(store) == 0 and store.is_empty()
        assert store.names == []

    def test_enrol_and_match(self):
        store = IdentityStore(threshold=0.5)
        embedding = vec(1)
        store.enroll("alice", embedding)

        match = store.match(embedding)

        assert match.matched and match.name == "alice"
        assert match.similarity == pytest.approx(1.0, abs=1e-5)
        assert match.label == "alice"

    def test_a_stranger_does_not_match(self):
        store = IdentityStore(threshold=0.9)
        store.enroll("alice", vec(1))

        match = store.match(vec(2))

        assert not match.matched
        assert match.label == "unknown"
        # The score is still reported, so "nearly matched" is distinguishable.
        assert -1.0 <= match.similarity <= 1.0

    def test_several_samples_per_identity(self):
        store = IdentityStore(threshold=0.9)
        store.enroll("alice", vec(1))
        store.enroll("alice", vec(2))

        assert len(store) == 1
        assert store.get("alice").sample_count == 2
        # Best-of, not an average: both poses match.
        assert store.match(vec(1)).matched
        assert store.match(vec(2)).matched

    def test_matching_is_best_of_all_identities(self):
        store = IdentityStore(threshold=0.9)
        store.enroll("alice", vec(1))
        store.enroll("bob", vec(2))

        assert store.match(vec(2)).name == "bob"

    def test_empty_store_matches_nothing(self):
        match = IdentityStore().match(vec(1))
        assert not match.matched and match.name == ""

    def test_removal_is_first_class(self):
        """The operation a data subject is most likely to ask for."""
        store = IdentityStore()
        store.enroll("alice", vec(1))

        assert store.remove("alice") is True
        assert store.is_empty()
        assert store.remove("alice") is False

    def test_rejects_empty_name(self):
        with pytest.raises(IdentityStoreError, match="must not be empty"):
            IdentityStore().enroll("   ", vec(1))

    def test_rejects_empty_embedding(self):
        with pytest.raises(IdentityStoreError, match="is empty"):
            IdentityStore().enroll("alice", np.array([], dtype=np.float32))

    def test_rejects_non_finite_embedding(self):
        bad = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        with pytest.raises(IdentityStoreError, match="NaN or inf"):
            IdentityStore().enroll("alice", bad)

    def test_rejects_a_mismatched_dimension(self):
        """Guards against silently comparing against a different face model."""
        store = IdentityStore()
        store.enroll("alice", vec(1, dim=8))

        with pytest.raises(IdentityStoreError, match="different face model"):
            store.enroll("bob", vec(2, dim=16))

    def test_describe_reports_counts_not_vectors(self):
        store = IdentityStore()
        store.enroll("alice", vec(1))

        described = store.describe()

        assert described["identities"] == 1 and described["samples"] == 1
        assert "embeddings" not in described


class TestIdentityPersistence:
    def test_round_trip(self, tmp_path):
        store = IdentityStore(threshold=0.5)
        store.enroll("alice", vec(1))
        store.enroll("alice", vec(2))
        store.enroll("bob", vec(3))
        path = store.save(tmp_path / "ids.npz")

        loaded = IdentityStore.load(path, threshold=0.5)

        assert loaded.names == ["alice", "bob"]
        assert loaded.get("alice").sample_count == 2
        assert loaded.match(vec(3)).name == "bob"

    def test_saved_file_contains_no_image_data(self, tmp_path):
        """The core privacy property: vectors, never pictures."""
        store = IdentityStore()
        store.enroll("alice", vec(1))
        path = store.save(tmp_path / "ids.npz")

        with zipfile.ZipFile(path) as archive:
            members = sorted(n.replace(".npy", "") for n in archive.namelist())

        assert members == ["embeddings", "enrolled_at", "names", "version"]

    def test_load_does_not_allow_pickle(self, tmp_path):
        """A store path comes from config; pickle on load would be RCE."""
        store = IdentityStore()
        store.enroll("alice", vec(1))
        path = store.save(tmp_path / "ids.npz")

        # Succeeds only because nothing in the file needs pickle.
        with np.load(path, allow_pickle=False) as data:
            assert data["names"].dtype.kind == "U"
            assert data["embeddings"].dtype == np.float32

    def test_empty_store_round_trips(self, tmp_path):
        path = IdentityStore().save(tmp_path / "empty.npz")
        assert IdentityStore.load(path).is_empty()

    def test_missing_file_is_actionable(self, tmp_path):
        with pytest.raises(IdentityStoreError, match="Enrol identities first"):
            IdentityStore.load(tmp_path / "nope.npz")

    def test_corrupt_file_is_reported(self, tmp_path):
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"not an npz")

        with pytest.raises(IdentityStoreError, match="Could not read"):
            IdentityStore.load(bad)

    def test_version_mismatch_is_reported(self, tmp_path):
        path = tmp_path / "old.npz"
        np.savez_compressed(
            path,
            version=np.array([99], dtype=np.int32),
            names=np.array(["alice"], dtype=np.str_),
            enrolled_at=np.array([""], dtype=np.str_),
            embeddings=np.zeros((1, DIM), dtype=np.float32),
        )

        with pytest.raises(IdentityStoreError, match="version 99"):
            IdentityStore.load(path)


def test_cosine_similarity_basics():
    a = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(a, np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.zeros(2)) == 0.0


# --------------------------------------------------------------------------- #
# Face detection
# --------------------------------------------------------------------------- #

def build_detector(**params) -> FaceDetectionModule:
    params.setdefault("backend", "fake")
    module = FaceDetectionModule(params)
    module.initialize()
    return module


class TestFaceDetection:
    def test_is_a_recognition_module_with_no_dependencies(self):
        assert FaceDetectionModule.module_type is ModuleType.RECOGNITION
        assert FaceDetectionModule.requires == ()

    def test_publishes_faces(self):
        FakeFaceDetector.scripted = [a_face(100), a_face(300)]
        module = build_detector()
        ctx = frame_ctx()

        result = module.process(ctx)

        faces = ctx.get_detections("face_detection")
        assert len(faces) == 2
        assert all(f.label == "face" for f in faces)
        assert result.data["count"] == 2

    def test_produces_no_identity(self):
        """Detection alone must never say who someone is."""
        FakeFaceDetector.scripted = [a_face()]
        module = build_detector()
        ctx = frame_ctx()

        module.process(ctx)

        face = ctx.get_detections("face_detection")[0]
        assert face.label == "face"
        assert "identity" not in face.extra

    def test_low_confidence_is_dropped(self):
        FakeFaceDetector.scripted = [a_face(conf=0.9), a_face(300, conf=0.2)]
        module = build_detector(confidence_threshold=0.5)
        ctx = frame_ctx()

        module.process(ctx)

        assert len(ctx.get_detections("face_detection")) == 1

    def test_tiny_faces_are_dropped(self):
        tiny = RawFace(confidence=0.9, xyxy=(10, 10, 20, 20))
        FakeFaceDetector.scripted = [tiny]
        module = build_detector(min_face_size=40)
        ctx = frame_ctx()

        module.process(ctx)

        assert ctx.get_detections("face_detection") == []

    def test_result_payload_is_json_safe(self):
        """Landmark arrays must not leak into anything serialised."""
        import json

        FakeFaceDetector.scripted = [a_face()]
        module = build_detector()
        ctx = frame_ctx()

        result = module.process(ctx)

        json.dumps(result.to_dict())
        assert "extra" not in result.detections[0]

    def test_backend_failure_is_a_module_error(self):
        FakeFaceDetector.fail_on_load = True
        with pytest.raises(ModuleInitializationError, match="no face model"):
            build_detector()

    def test_unknown_backend_is_reported(self):
        with pytest.raises(ModuleInitializationError, match="Unknown face detector"):
            FaceDetectionModule({"backend": "telepathy"}).initialize()

    def test_cleanup_unloads(self):
        module = build_detector()
        module.cleanup()
        assert module._backend is None


# --------------------------------------------------------------------------- #
# Face recognition
# --------------------------------------------------------------------------- #

def a_store(tmp_path, **people) -> str:
    store = IdentityStore(threshold=0.5)
    for name, embedding in people.items():
        store.enroll(name, embedding)
    return str(store.save(tmp_path / "ids.npz"))


def build_recogniser(tmp_path, people=None, **params) -> FaceRecognitionModule:
    params.setdefault("backend", "fake")
    params.setdefault("identity_store", a_store(tmp_path, **(people or {})))
    module = FaceRecognitionModule(params)
    module.initialize()
    return module


def ctx_with_faces(*faces) -> FrameContext:
    ctx = frame_ctx()
    ctx.add_detections(
        "face_detection",
        [
            Detection(
                label="face",
                confidence=f.confidence,
                bbox=BBox.from_xyxy(f.xyxy),
                extra={"face": f},
            )
            for f in faces
        ],
    )
    return ctx


class TestFaceRecognition:
    def test_requires_face_detection(self):
        assert FaceRecognitionModule.requires == ("face_detection",)

    def test_refuses_to_start_without_an_identity_store(self):
        """Better than silently labelling everybody unknown."""
        with pytest.raises(ModuleInitializationError, match="needs an identity store"):
            FaceRecognitionModule({"backend": "fake"}).initialize()

    def test_missing_store_file_is_actionable(self, tmp_path):
        with pytest.raises(ModuleInitializationError, match="Enrol identities first"):
            FaceRecognitionModule(
                {"backend": "fake", "identity_store": str(tmp_path / "nope.npz")}
            ).initialize()

    def test_recognises_an_enrolled_face(self, tmp_path):
        alice = vec(1)
        FakeFaceEmbedder.by_x = {100: alice}
        module = build_recogniser(tmp_path, people={"alice": alice})
        ctx = ctx_with_faces(a_face(100))

        result = module.process(ctx)

        detections = ctx.get_detections("face_recognition")
        assert [d.label for d in detections] == ["alice"]
        assert detections[0].extra["recognised"] is True
        assert result.data["recognised"] == 1

    def test_reports_an_unenrolled_face_as_unknown(self, tmp_path):
        FakeFaceEmbedder.by_x = {100: vec(42)}
        module = build_recogniser(tmp_path, people={"alice": vec(1)})
        ctx = ctx_with_faces(a_face(100))

        result = module.process(ctx)

        detection = ctx.get_detections("face_recognition")[0]
        assert detection.label == "unknown"
        assert detection.extra["recognised"] is False
        assert result.data["unknown"] == 1

    def test_never_enrols_implicitly(self, tmp_path):
        """Seeing a face must never add it to the store."""
        FakeFaceEmbedder.by_x = {100: vec(42)}
        module = build_recogniser(tmp_path, people={"alice": vec(1)})

        for frame in range(5):
            module.process(ctx_with_faces(a_face(100)))

        assert module.enrolled_names == ["alice"]

    def test_tells_the_output_layer_it_consumed_the_detector(self, tmp_path):
        FakeFaceEmbedder.by_x = {100: vec(1)}
        module = build_recogniser(tmp_path, people={"alice": vec(1)})

        result = module.process(ctx_with_faces(a_face(100)))

        assert result.data["consumed_from"] == "face_detection"

    def test_distinguishes_two_people(self, tmp_path):
        alice, bob = vec(1), vec(2)
        FakeFaceEmbedder.by_x = {100: alice, 300: bob}
        module = build_recogniser(tmp_path, people={"alice": alice, "bob": bob})
        ctx = ctx_with_faces(a_face(100), a_face(300))

        module.process(ctx)

        assert {d.label for d in ctx.get_detections("face_recognition")} == {"alice", "bob"}

    def test_max_faces_caps_work(self, tmp_path):
        FakeFaceEmbedder.by_x = {}
        module = build_recogniser(tmp_path, people={"alice": vec(1)}, max_faces=2)
        ctx = ctx_with_faces(a_face(100), a_face(200), a_face(300), a_face(400))

        module.process(ctx)

        assert len(ctx.get_detections("face_recognition")) == 2

    def test_an_empty_store_warns_and_reports_unknown(self, tmp_path, caplog):
        FakeFaceEmbedder.by_x = {100: vec(1)}
        with caplog.at_level("WARNING"):
            module = build_recogniser(tmp_path)

        ctx = ctx_with_faces(a_face(100))
        module.process(ctx)

        assert "no identities" in caplog.text
        assert ctx.get_detections("face_recognition")[0].label == "unknown"

    def test_a_bad_crop_does_not_break_the_frame(self, tmp_path, monkeypatch):
        module = build_recogniser(tmp_path, people={"alice": vec(1)})
        monkeypatch.setattr(
            module._embedder,
            "embed",
            lambda frame, face: (_ for _ in ()).throw(ValueError("bad crop")),
        )
        ctx = ctx_with_faces(a_face(100))

        result = module.process(ctx)

        assert result.ok
        assert result.data["count"] == 0

    def test_cleanup_drops_the_store_from_memory(self, tmp_path):
        module = build_recogniser(tmp_path, people={"alice": vec(1)})
        module.cleanup()
        assert module._store is None
        assert module.enrolled_names == []


# --------------------------------------------------------------------------- #
# Unknown person
# --------------------------------------------------------------------------- #

def unknown_ctx(count: int = 1, ts: float = 0.0, similarity: float = 0.1) -> FrameContext:
    ctx = frame_ctx(ts=ts)
    ctx.add_detections(
        "face_recognition",
        [
            Detection(
                label="unknown",
                confidence=0.9,
                bbox=BBox(100 + i * 80, 50, 160 + i * 80, 120),
                extra={"recognised": False, "similarity": similarity},
            )
            for i in range(count)
        ],
    )
    return ctx


class TestUnknownPerson:
    def test_is_analysis_requiring_recognition(self):
        assert UnknownPersonModule.module_type is ModuleType.ANALYSIS
        assert UnknownPersonModule.requires == ("face_recognition",)

    def test_loads_no_model(self):
        module = UnknownPersonModule({})
        module.initialize()
        assert not hasattr(module, "_backend")

    def test_reports_after_min_sightings(self):
        module = UnknownPersonModule({"min_sightings": 3})
        module.initialize()

        fired = []
        for frame in range(4):
            ctx = unknown_ctx(ts=frame * 0.1)
            module.process(ctx)
            fired.append(len(ctx.events))

        assert fired == [0, 0, 1, 0]  # third frame reports; cooldown then quiets

    def test_a_single_glimpse_is_not_reported(self):
        """Recognition is least reliable exactly when it briefly fails."""
        module = UnknownPersonModule({"min_sightings": 3})
        module.initialize()

        ctx = unknown_ctx()
        module.process(ctx)

        assert ctx.events == []

    def test_a_recognised_face_resets_the_streak(self):
        module = UnknownPersonModule({"min_sightings": 2})
        module.initialize()
        module.process(unknown_ctx(ts=0.0))

        known = frame_ctx(ts=0.1)
        known.add_detections(
            "face_recognition",
            [
                Detection(
                    label="alice",
                    confidence=0.9,
                    bbox=BBox(100, 50, 160, 120),
                    extra={"recognised": True, "similarity": 0.9},
                )
            ],
        )
        module.process(known)

        assert known.events == []
        following = unknown_ctx(ts=0.2)
        module.process(following)
        assert following.events == []  # streak restarted

    def test_cooldown_prevents_an_event_per_frame(self):
        """Without this a live camera makes thousands of events an hour."""
        module = UnknownPersonModule({"min_sightings": 1, "cooldown_seconds": 10})
        module.initialize()

        total = 0
        for frame in range(30):
            ctx = unknown_ctx(ts=frame * 0.1)  # 3 seconds of footage
            module.process(ctx)
            total += len(ctx.events)

        assert total == 1

    def test_cooldown_expires(self):
        module = UnknownPersonModule({"min_sightings": 1, "cooldown_seconds": 1.0})
        module.initialize()
        module.process(unknown_ctx(ts=0.0))

        later = unknown_ctx(ts=2.0)
        module.process(later)

        assert len(later.events) == 1

    def test_event_carries_the_closest_similarity(self):
        """Distinguishes 'nobody like them' from 'almost matched someone'."""
        module = UnknownPersonModule({"min_sightings": 1})
        module.initialize()
        ctx = unknown_ctx(similarity=0.34)

        module.process(ctx)

        event = ctx.events[0]
        assert event.event_type == EVENT_UNKNOWN_PERSON
        assert event.data["closest_similarity"] == 0.34
        assert event.severity is Severity.WARNING

    def test_severity_is_configurable(self):
        module = UnknownPersonModule({"min_sightings": 1, "severity": "critical"})
        module.initialize()
        ctx = unknown_ctx()

        module.process(ctx)

        assert ctx.events[0].severity is Severity.CRITICAL

    def test_bad_severity_is_rejected(self):
        with pytest.raises(ValueError, match="severity"):
            UnknownPersonModule({"severity": "spooky"})

    def test_no_faces_means_no_events(self):
        module = UnknownPersonModule({"min_sightings": 1})
        module.initialize()
        ctx = frame_ctx()

        result = module.process(ctx)

        assert ctx.events == []
        assert result.data["unknown_faces"] == 0


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

class TestWiring:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("face_detection", FaceDetectionModule),
            ("face_recognition", FaceRecognitionModule),
            ("unknown_person", UnknownPersonModule),
        ],
    )
    def test_resolves_by_name(self, name, cls):
        assert ModuleRegistry.resolve(name) is cls

    def test_identity_chain_is_ordered(self, tmp_path):
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        # Scrambled on purpose.
                        "unknown_person": {"enabled": True},
                        "face_recognition": {
                            "enabled": True,
                            "backend": "fake",
                            "identity_store": a_store(tmp_path, alice=vec(1)),
                        },
                        "face_detection": {"enabled": True, "backend": "fake"},
                    }
                }
            )
        )
        order = [m.module_name for m in built]
        assert order == ["face_detection", "face_recognition", "unknown_person"]

    def test_recognition_without_detection_is_rejected(self, tmp_path):
        with pytest.raises(ModuleDependencyError, match="face_detection"):
            ModuleRegistry.build_enabled(
                build_config(
                    {
                        "modules": {
                            "face_recognition": {
                                "enabled": True,
                                "backend": "fake",
                                "identity_store": a_store(tmp_path, alice=vec(1)),
                            }
                        }
                    }
                )
            )

    def test_unknown_person_without_recognition_is_rejected(self):
        with pytest.raises(ModuleDependencyError, match="face_recognition"):
            ModuleRegistry.build_enabled(
                build_config({"modules": {"unknown_person": {"enabled": True}}})
            )

    def test_all_identity_modules_are_disabled_in_the_shipped_config(self):
        """Agent.md §24: identity processing is opt-in, never on by default."""
        from computer_vision.core.config import load_config

        config = load_config("configs/config.yaml", use_env=False)

        for name in ("face_detection", "face_recognition", "unknown_person"):
            assert config.modules[name].enabled is False

    def test_face_detection_alone_needs_no_identity_store(self):
        """Finding faces must not require enrolling anyone."""
        built = ModuleRegistry.build_enabled(
            build_config(
                {"modules": {"face_detection": {"enabled": True, "backend": "fake"}}}
            )
        )
        assert [m.module_name for m in built] == ["face_detection"]


class TestFullChain:
    def test_detection_to_recognition_to_event(self, tmp_path):
        """A stranger walks in front of the camera and one event results."""
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import VisionPipeline
        from tools.tests.test_pipeline import RecordingSink, SyntheticSource

        FakeFaceDetector.scripted = [a_face(100)]
        FakeFaceEmbedder.by_x = {100: vec(42)}  # nothing like alice

        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "face_detection": {"enabled": True, "backend": "fake"},
                    "face_recognition": {
                        "enabled": True,
                        "backend": "fake",
                        "identity_store": a_store(tmp_path, alice=vec(1)),
                    },
                    "unknown_person": {"enabled": True, "min_sightings": 2},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        sink = RecordingSink()
        VisionPipeline(
            config=config,
            source=SyntheticSource(frames=5),
            modules=modules,
            output=sink,
            processor=FrameProcessor(config.processing),
        ).run()

        events = [e for c in sink.contexts for e in c.events]
        assert [e.event_type for e in events] == [EVENT_UNKNOWN_PERSON]

    def test_an_enrolled_person_raises_nothing(self, tmp_path):
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import VisionPipeline
        from tools.tests.test_pipeline import RecordingSink, SyntheticSource

        alice = vec(1)
        FakeFaceDetector.scripted = [a_face(100)]
        FakeFaceEmbedder.by_x = {100: alice}

        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "face_detection": {"enabled": True, "backend": "fake"},
                    "face_recognition": {
                        "enabled": True,
                        "backend": "fake",
                        "identity_store": a_store(tmp_path, alice=alice),
                    },
                    "unknown_person": {"enabled": True, "min_sightings": 1},
                },
            }
        )
        modules = ModuleRegistry.build_enabled(config)
        sink = RecordingSink()
        VisionPipeline(
            config=config,
            source=SyntheticSource(frames=5),
            modules=modules,
            output=sink,
            processor=FrameProcessor(config.processing),
        ).run()

        assert [e for c in sink.contexts for e in c.events] == []
        assert sink.contexts[-1].get_detections("face_recognition")[0].label == "alice"
