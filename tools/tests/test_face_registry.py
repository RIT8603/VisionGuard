"""Face registry: users, multi-angle enrolment, live store, API.

Runs without any model file: the service is given a fake detector and embedder
that report whatever the test wants, so the rules under test (pose gating,
quality gating, duplicate detection, live-store rebuilds) are exercised on
their own and never depend on a real face.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.persistence.face_db import (
    DuplicateUserError,
    FaceDatabase,
    FaceDbError,
    UserNotFoundError,
    clean_fields,
)
from backend.services.faces import (
    MAX_SAMPLES_PER_POSE,
    FaceService,
    FaceServiceError,
    classify_pose,
    estimate_pose_angles,
)
from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.identity.shared import get_shared_store, replace_shared_store
from computer_vision.identity.store import IdentityStore
from computer_vision.inference.face_base import RawFace

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

#: Landmarks (right eye, left eye, nose, right mouth, left mouth) for each view.
#: Eyes sit 100 px apart at y=170, the mouth at y=270; only the nose moves.
EYES_MOUTH = ([150, 170], [250, 170]), ([160, 270], [240, 270])
NOSE = {
    "front": [200, 223],  # yaw 0,    pitch 0.53
    "left": [230, 223],   # yaw +0.3
    "right": [170, 223],  # yaw -0.3
    "up": [200, 205],     # pitch 0.35
    "down": [200, 242],   # pitch 0.72
}


def landmarks(view: str) -> np.ndarray:
    (re, le), (rm, lm) = EYES_MOUTH
    return np.array([re, le, NOSE[view], rm, lm], dtype=np.float32)


class FakeDetector:
    def __init__(self) -> None:
        self.faces: list[RawFace] = []

    def show(self, view: str, *, confidence: float = 0.95, box=(100, 100, 300, 300)) -> None:
        self.faces = [RawFace(confidence, tuple(map(float, box)), landmarks(view))]

    def detect(self, frame):
        return list(self.faces)


class FakeEmbedder:
    def __init__(self) -> None:
        self.vector = np.zeros(128, dtype=np.float32)

    def embed(self, frame, face):
        return self.vector.copy()


def unit(seed: int, noise: float = 0.0, base: int | None = None) -> np.ndarray:
    """A deterministic 128-d vector; vectors with different seeds are near-orthogonal."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(128).astype(np.float32)
    if base is not None:
        vec = np.random.default_rng(base).standard_normal(128).astype(np.float32)
        vec = vec + noise * rng.standard_normal(128).astype(np.float32)
    return vec / np.linalg.norm(vec)


def jpeg(sharp: bool = True) -> bytes:
    """A 400x400 image: noise is 'sharp', a flat grey is blurry."""
    if sharp:
        rng = np.random.default_rng(1)
        image = rng.integers(60, 200, (400, 400, 3), dtype=np.uint8)
    else:
        image = np.full((400, 400, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def clean_store():
    replace_shared_store([])
    yield
    replace_shared_store([])


@pytest.fixture
def svc(tmp_path):
    service = FaceService(FaceDatabase(tmp_path / "faces.db"))
    service._detector = FakeDetector()
    service._embedder = FakeEmbedder()
    return service


def enrol(svc: FaceService, user_id: int, view: str, vector: np.ndarray):
    svc._detector.show(view)
    svc._embedder.vector = vector
    return svc.add_sample(user_id, view, jpeg())


# --------------------------------------------------------------------------- #
# Pose maths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("view", ["front", "left", "right", "up", "down"])
def test_each_view_is_classified_as_itself(view):
    yaw, pitch, _ = estimate_pose_angles(landmarks(view))
    assert classify_pose(yaw, pitch) == view


def test_full_profile_is_not_a_left_or_right_capture():
    assert classify_pose(1.4, 0.53) == "other"
    assert classify_pose(-1.4, 0.53) == "other"
    assert classify_pose(0.7, 0.53) == "left"


def test_between_two_views_is_other():
    # Turned a little: past 'front', short of 'left'.
    assert classify_pose(0.17, 0.53) == "other"


def test_degenerate_landmarks_do_not_divide_by_zero():
    flat = np.zeros((5, 2), dtype=np.float32)
    assert estimate_pose_angles(flat) == (0.0, 0.5, 0.0)


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

def test_user_names_are_unique_ignoring_case(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    db.create_user({"name": "Asha Rao"})
    with pytest.raises(DuplicateUserError):
        db.create_user({"name": "asha rao"})


def test_deleting_a_user_deletes_their_templates(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    user = db.create_user({"name": "Asha"})
    db.add_sample(user.id, "front", unit(1), 0.9)
    assert db.active_identities()
    db.delete_user(user.id)
    assert db.active_identities() == []
    assert db.embedding_dim() is None


def test_missing_user_is_reported(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    with pytest.raises(UserNotFoundError):
        db.get_user(99)


@pytest.mark.parametrize(
    "body",
    [{}, {"name": "   "}, {"name": 5}, {"name": "x" * 200}, {"name": "ok", "notes": "n" * 600}],
)
def test_bad_user_fields_are_rejected(body):
    with pytest.raises(FaceDbError):
        clean_fields(body, require_name=True)


def test_text_fields_are_trimmed_and_collapsed():
    assert clean_fields({"name": "  Asha   Rao "}, require_name=True)["name"] == "Asha Rao"


# --------------------------------------------------------------------------- #
# Enrolment rules
# --------------------------------------------------------------------------- #

def test_enrolling_all_three_required_views_completes_a_user(svc):
    user = svc.create_user({"name": "Asha"})
    assert user["enrolled"] is False
    for view in ("front", "left", "right"):
        result = enrol(svc, user["id"], view, unit(1, 0.2, base=1))
    assert result["user"]["enrolled"] is True
    assert result["user"]["missing_poses"] == []


def test_wrong_pose_is_refused_with_a_hint(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.show("left")
    with pytest.raises(FaceServiceError, match="does not look like the 'front' view"):
        svc.add_sample(user["id"], "front", jpeg())
    assert svc.get_user(user["id"])["sample_count"] == 0


def test_auto_pose_files_the_photo_under_what_it_shows(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.show("right")
    svc._embedder.vector = unit(1)
    assert svc.add_sample(user["id"], "auto", jpeg())["pose"] == "right"


def test_blurry_image_is_refused(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.show("front")
    with pytest.raises(FaceServiceError, match="blurry"):
        svc.add_sample(user["id"], "front", jpeg(sharp=False))


def test_small_face_is_refused(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.show("front", box=(150, 150, 200, 200))
    with pytest.raises(FaceServiceError, match="too small"):
        svc.add_sample(user["id"], "front", jpeg())


def test_face_cut_off_by_the_frame_is_refused(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.show("front", box=(100, 100, 405, 300))
    with pytest.raises(FaceServiceError, match="inside the frame"):
        svc.add_sample(user["id"], "front", jpeg())


def test_two_faces_are_refused(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.show("front")
    svc._detector.faces.append(RawFace(0.95, (10.0, 10.0, 170.0, 170.0), landmarks("front")))
    with pytest.raises(FaceServiceError, match="More than one face"):
        svc.add_sample(user["id"], "front", jpeg())


def test_no_face_is_refused(svc):
    user = svc.create_user({"name": "Asha"})
    svc._detector.faces = []
    with pytest.raises(FaceServiceError, match="No face found"):
        svc.add_sample(user["id"], "front", jpeg())


def test_unreadable_upload_is_refused(svc):
    user = svc.create_user({"name": "Asha"})
    with pytest.raises(FaceServiceError, match="not a readable"):
        svc.add_sample(user["id"], "front", b"not an image")


def test_another_persons_face_cannot_be_added_to_a_user(svc):
    user = svc.create_user({"name": "Asha"})
    enrol(svc, user["id"], "front", unit(1))
    with pytest.raises(FaceServiceError, match="does not match the earlier captures") as exc:
        enrol(svc, user["id"], "left", unit(2))  # unrelated vector
    assert exc.value.status == 409


def test_same_face_cannot_be_registered_under_two_names(svc):
    asha = svc.create_user({"name": "Asha"})
    ravi = svc.create_user({"name": "Ravi"})
    enrol(svc, asha["id"], "front", unit(1))
    with pytest.raises(FaceServiceError, match="already registered as 'Asha'") as exc:
        enrol(svc, ravi["id"], "front", unit(1, 0.1, base=1))
    assert exc.value.status == 409


def test_pose_capture_limit(svc):
    user = svc.create_user({"name": "Asha"})
    for _ in range(MAX_SAMPLES_PER_POSE):
        enrol(svc, user["id"], "front", unit(1, 0.1, base=1))
    with pytest.raises(FaceServiceError, match="clear it to retake"):
        enrol(svc, user["id"], "front", unit(1, 0.1, base=1))


def test_clearing_one_pose_leaves_the_others(svc):
    user = svc.create_user({"name": "Asha"})
    enrol(svc, user["id"], "front", unit(1))
    enrol(svc, user["id"], "left", unit(1, 0.1, base=1))
    result = svc.clear_samples(user["id"], "left")
    assert result["poses"] == {"front": 1}


def test_read_only_registry_refuses_changes(tmp_path):
    service = FaceService(FaceDatabase(tmp_path / "f.db"), editable=False)
    with pytest.raises(FaceServiceError) as exc:
        service.create_user({"name": "Asha"})
    assert exc.value.status == 403


# --------------------------------------------------------------------------- #
# The live store follows the database
# --------------------------------------------------------------------------- #

def test_enrolled_user_is_recognised_immediately(svc):
    user = svc.create_user({"name": "Asha"})
    face = unit(1)
    assert get_shared_store().match(face).matched is False
    enrol(svc, user["id"], "front", face)
    match = get_shared_store().match(face)
    assert match.matched and match.name == "Asha"


def test_deleting_or_deactivating_a_user_stops_recognition(svc):
    user = svc.create_user({"name": "Asha"})
    face = unit(1)
    enrol(svc, user["id"], "front", face)

    svc.update_user(user["id"], {"active": False})
    assert get_shared_store().match(face).matched is False

    svc.update_user(user["id"], {"active": True})
    assert get_shared_store().match(face).matched is True

    svc.delete_user(user["id"])
    assert get_shared_store().match(face).matched is False


def test_renaming_a_user_renames_the_recognised_label(svc):
    user = svc.create_user({"name": "Asha"})
    face = unit(1)
    enrol(svc, user["id"], "front", face)
    svc.update_user(user["id"], {"name": "Asha Rao"})
    assert get_shared_store().match(face).name == "Asha Rao"


def test_store_is_rebuilt_from_disk_on_restart(tmp_path):
    db_path = tmp_path / "f.db"
    first = FaceService(FaceDatabase(db_path))
    first._detector, first._embedder = FakeDetector(), FakeEmbedder()
    user = first.create_user({"name": "Asha"})
    enrol(first, user["id"], "front", unit(1))

    replace_shared_store([])  # a new process starts with an empty store
    FaceService(FaceDatabase(db_path))
    assert get_shared_store().match(unit(1)).name == "Asha"


# --------------------------------------------------------------------------- #
# Matching accuracy rules
# --------------------------------------------------------------------------- #

def test_look_alike_is_not_named_when_two_people_score_close():
    store = IdentityStore(threshold=0.3)
    base = unit(1)
    twin = base + 0.05 * unit(2)
    store.enroll("A", base)
    store.enroll("B", twin / np.linalg.norm(twin))
    probe = base
    assert store.match(probe).matched is True             # no margin: picks one
    assert store.match(probe, margin=0.2).matched is False  # ambiguous: refuses


def test_margin_does_not_block_a_clear_winner():
    store = IdentityStore(threshold=0.3)
    store.enroll("A", unit(1))
    store.enroll("B", unit(2))
    assert store.match(unit(1), margin=0.2).name == "A"


# --------------------------------------------------------------------------- #
# Recognition module: quality gate, best snapshot, person-track lock
# --------------------------------------------------------------------------- #

from computer_vision.core.context import Track  # noqa: E402


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed(self, frame, face):
        self.calls += 1
        return super().embed(frame, face)


def make_module(embedder: FakeEmbedder, **params):
    from computer_vision.modules.recognition.face_recognition import FaceRecognitionModule

    module = FaceRecognitionModule({"margin": 0.0, "embed_interval": 1, **params})
    module._embedder = embedder
    module._store = get_shared_store()
    return module


_SHARP = np.random.default_rng(3).integers(40, 220, (500, 500, 3), dtype=np.uint8)
_FLAT = np.full((500, 500, 3), 128, dtype=np.uint8)
BOX = (100, 100, 220, 240)  # 120 px wide: a good-sized face
SMALL = (100, 100, 150, 150)  # a far-away, lower-scoring face
BODY = (40, 40, 330, 490)  # a person box containing every face box above
LOOSE = {"min_quality": 0.0, "min_sharpness": 0, "min_face_size": 0}


def frame_with_face(frame_id, box=BOX, *, view="front", frame=_SHARP, persons=(), faces=True):
    ctx = FrameContext(
        frame=frame, frame_id=frame_id, source_id="t", monotonic_ts=float(frame_id)
    )
    if faces:
        raw = RawFace(0.95, tuple(map(float, box)), landmarks(view))
        ctx.add_detections(
            "face_detection",
            [Detection("face", 0.95, BBox(*box), source_module="face_detection", extra={"face": raw})],
        )
    if persons:
        ctx.add_tracks(
            "person_tracking",
            [Track(track_id=tid, label="person", bbox=BBox(*pbox)) for tid, pbox in persons],
        )
    return ctx


def run(module, ctx):
    module.process(ctx)
    dets = ctx.get_detections("face_recognition")
    return dets[0] if dets else None


def test_module_labels_a_registered_face_by_name():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = FakeEmbedder()
    embedder.vector = unit(1)
    det = run(make_module(embedder), frame_with_face(1))
    assert det.label == "Asha" and det.extra["recognised"] is True


def test_module_reports_a_checked_stranger_as_unknown():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = FakeEmbedder()
    embedder.vector = unit(7)
    det = run(make_module(embedder), frame_with_face(1))
    assert det.label == "unknown" and det.extra["verified"] is True


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"frame": _FLAT}, "blurry"),
        ({"box": (100, 100, 130, 128)}, "small"),
    ],
)
def test_unreadable_faces_are_not_embedded(kwargs, reason):
    replace_shared_store([("Asha", [unit(1)])])
    embedder = CountingEmbedder()
    embedder.vector = unit(1)
    det = run(make_module(embedder), frame_with_face(1, **kwargs))
    assert embedder.calls == 0
    assert reason in det.extra["skipped"]
    # Not a finding: an unreadable face is not "an unknown person".
    assert det.label == "face" and det.extra["verified"] is False


def test_a_face_turned_too_far_is_not_embedded():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = CountingEmbedder()
    embedder.vector = unit(1)
    ctx = frame_with_face(1)
    # Nose pushed far to one side: yaw ratio well past the 0.5 limit.
    ctx.get_detections("face_detection")[0].extra["face"].landmarks[2] = [260, 223]
    det = run(make_module(embedder), ctx)
    assert embedder.calls == 0 and "turned" in det.extra["skipped"]


def test_unknown_person_ignores_unreadable_faces():
    from computer_vision.modules.recognition.unknown_person import UnknownPersonModule

    replace_shared_store([("Asha", [unit(1)])])
    module = make_module(FakeEmbedder())
    ctx = frame_with_face(1, frame=_FLAT)
    module.process(ctx)
    result = UnknownPersonModule({"min_sightings": 1}).process(ctx)
    assert result.data["unknown_faces"] == 0


def test_a_known_name_is_kept_when_a_worse_frame_cannot_say():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = CountingEmbedder()
    module = make_module(embedder)

    embedder.vector = unit(1)
    assert run(module, frame_with_face(1)).label == "Asha"
    calls = embedder.calls

    # Same-quality frames do not re-run the recogniser at all.
    for frame_id in range(2, 6):
        assert run(module, frame_with_face(frame_id)).label == "Asha"
    assert embedder.calls == calls

    # A blurred frame cannot erase the name.
    embedder.vector = unit(9)
    assert run(module, frame_with_face(6, frame=_FLAT)).label == "Asha"


def test_a_clearly_better_snapshot_is_checked_again():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = CountingEmbedder()
    module = make_module(embedder, **LOOSE)
    embedder.vector = unit(1)

    # The same person (one body track) first far away, then much closer.
    assert run(module, frame_with_face(1, box=SMALL, persons=[(1, BODY)])).label == "Asha"
    before = embedder.calls
    run(module, frame_with_face(2, box=BOX, persons=[(1, BODY)]))  # a better snapshot
    assert embedder.calls == before + 1
    # And a snapshot no better than that one is not checked again.
    run(module, frame_with_face(3, box=BOX, persons=[(1, BODY)]))
    assert embedder.calls == before + 1


def test_replacing_a_name_takes_repeated_better_evidence():
    replace_shared_store([("Asha", [unit(1)]), ("Ravi", [unit(2)])])
    embedder = CountingEmbedder()
    module = make_module(embedder, **LOOSE)

    embedder.vector = unit(1)
    assert run(module, frame_with_face(1, box=SMALL, persons=[(1, BODY)])).label == "Asha"

    # One better frame that says "Ravi" is not enough to rename the subject...
    embedder.vector = unit(2)
    assert run(module, frame_with_face(2, box=BOX, persons=[(1, BODY)])).label == "Asha"
    # ...a second, better one is.
    assert run(module, frame_with_face(3, box=(100, 100, 300, 360), persons=[(1, BODY)])).label == "Ravi"


def test_name_follows_the_person_track_when_the_face_is_lost():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = FakeEmbedder()
    embedder.vector = unit(1)
    module = make_module(embedder)
    body = (60, 60, 300, 480)

    det = run(module, frame_with_face(1, persons=[(3, body)]))
    assert det.label == "Asha" and det.extra["person_track_id"] == 3

    # The person turns away: no face detected for a while. The subject lives on.
    for frame_id in range(2, 8):
        module.process(frame_with_face(frame_id, persons=[(3, body)], faces=False))

    # They come back side-on (too turned to recognise): still Asha.
    embedder.vector = unit(9)
    ctx = frame_with_face(8, persons=[(3, body)])
    ctx.get_detections("face_detection")[0].extra["face"].landmarks[2] = [260, 223]
    assert run(module, ctx).label == "Asha"


def test_two_people_keep_their_own_names():
    replace_shared_store([("Asha", [unit(1)]), ("Ravi", [unit(2)])])
    embedder = FakeEmbedder()
    module = make_module(embedder)

    ctx = FrameContext(frame=_SHARP, frame_id=1, source_id="t", monotonic_ts=1.0)
    faces = []
    for box in [(50, 50, 170, 190), (300, 50, 420, 190)]:
        raw = RawFace(0.95, tuple(map(float, box)), landmarks("front"))
        faces.append(
            Detection("face", 0.95, BBox(*box), source_module="face_detection", extra={"face": raw})
        )
    ctx.add_detections("face_detection", faces)
    ctx.add_tracks(
        "person_tracking",
        [Track(1, "person", BBox(30, 30, 200, 480)), Track(2, "person", BBox(280, 30, 450, 480))],
    )

    vectors = iter([unit(1), unit(2)])
    embedder.embed = lambda frame, face: next(vectors)  # type: ignore[method-assign]
    module.process(ctx)
    assert [d.label for d in ctx.get_detections("face_recognition")] == ["Asha", "Ravi"]


def test_face_outside_every_body_falls_back_to_face_tracking():
    replace_shared_store([("Asha", [unit(1)])])
    embedder = FakeEmbedder()
    embedder.vector = unit(1)
    module = make_module(embedder)
    det = run(module, frame_with_face(1, persons=[(9, (400, 300, 480, 490))]))
    assert det.label == "Asha" and "person_track_id" not in det.extra


def test_recognition_runs_after_person_tracking_without_requiring_it():
    from computer_vision.modules.recognition.face_recognition import FaceRecognitionModule

    assert "person_tracking" in FaceRecognitionModule.run_after
    assert "person_tracking" not in FaceRecognitionModule.requires


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #

fastapi = pytest.importorskip("fastapi", reason="dashboard extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.server import build_app  # noqa: E402
from backend.services.state import DashboardState  # noqa: E402


@pytest.fixture
def client(svc):
    app = build_app(DashboardState(max_events=5), faces=svc)
    return TestClient(app, base_url="http://127.0.0.1")


def test_full_registration_flow_over_http(client, svc):
    created = client.post("/api/face/users", json={"name": "Asha", "role": "Engineer"})
    assert created.status_code == 201
    user_id = created.json()["id"]

    svc._embedder.vector = unit(1)
    for view in ("front", "left", "right"):
        svc._detector.show(view)
        response = client.post(f"/api/face/users/{user_id}/samples?pose={view}", content=jpeg())
        assert response.status_code == 201, response.text

    assert client.get(f"/api/face/users/{user_id}").json()["enrolled"] is True
    assert client.get("/api/face/status").json()["enrolled"] == 1
    assert get_shared_store().match(unit(1)).name == "Asha"

    assert client.delete(f"/api/face/users/{user_id}").status_code == 200
    assert client.get("/api/face/users").json()["users"] == []
    assert get_shared_store().match(unit(1)).matched is False


def test_analyze_reports_pose_without_storing(client, svc):
    svc._detector.show("left")
    body = client.post("/api/face/analyze", content=jpeg()).json()
    assert body["ok"] is True and body["face"]["pose"] == "left"
    assert client.get("/api/face/status").json()["templates"] == 0


def test_http_errors_carry_a_status_and_message(client, svc):
    assert client.post("/api/face/users", json={}).status_code == 400
    client.post("/api/face/users", json={"name": "Asha"})
    assert client.post("/api/face/users", json={"name": "ASHA"}).status_code == 409
    assert client.get("/api/face/users/999").status_code == 404
    bad = client.post("/api/face/users/1/samples?pose=front", content=b"junk")
    assert bad.status_code == 400 and "readable" in bad.json()["error"]


def test_oversized_upload_is_refused(client):
    client.post("/api/face/users", json={"name": "Asha"})
    huge = b"0" * (8 * 1024 * 1024 + 1)
    assert client.post("/api/face/users/1/samples?pose=front", content=huge).status_code == 413


def test_routes_answer_503_without_a_registry():
    bare = TestClient(build_app(DashboardState(max_events=5)), base_url="http://127.0.0.1")
    assert bare.get("/api/face/users").status_code == 503
    assert bare.post("/api/face/users", json={"name": "x"}).status_code == 503


# --------------------------------------------------------------------------- #
# Testing recognition and estimating accuracy
# --------------------------------------------------------------------------- #

def test_identify_names_a_registered_face_and_ranks_candidates(svc):
    asha = svc.create_user({"name": "Asha"})
    ravi = svc.create_user({"name": "Ravi"})
    enrol(svc, asha["id"], "front", unit(1))
    enrol(svc, ravi["id"], "front", unit(2))
    svc._detector.show("front")
    svc._embedder.vector = unit(1, 0.1, base=1)
    result = svc.identify(jpeg())
    assert result["decision"] == "Asha"
    assert [m["name"] for m in result["matches"]] == ["Asha", "Ravi"]


def test_identify_says_unknown_for_a_stranger(svc):
    asha = svc.create_user({"name": "Asha"})
    enrol(svc, asha["id"], "front", unit(1))
    svc._detector.show("front")
    svc._embedder.vector = unit(9)
    result = svc.identify(jpeg())
    assert result["decision"] == "unknown" and "below the threshold" in result["reason"]


def test_accuracy_estimate_over_templates(svc):
    asha = svc.create_user({"name": "Asha"})
    ravi = svc.create_user({"name": "Ravi"})
    for view in ("front", "left"):
        enrol(svc, asha["id"], view, unit(1, 0.2, base=1))
        enrol(svc, ravi["id"], view, unit(2, 0.2, base=2))
    report = svc.accuracy()
    assert report["identities"] == 2 and report["trials"] == 4
    assert report["rank1_accuracy"] == 1.0
    assert report["false_accept_rate"] == 0.0
    assert report["suggested_threshold"] is not None


def test_identify_and_accuracy_over_http(client, svc):
    user = client.post("/api/face/users", json={"name": "Asha"}).json()
    svc._embedder.vector = unit(1)
    svc._detector.show("front")
    client.post(f"/api/face/users/{user['id']}/samples?pose=front", content=jpeg())
    assert client.post("/api/face/identify", content=jpeg()).json()["decision"] == "Asha"
    assert client.get("/api/face/accuracy").json()["identities"] == 1
