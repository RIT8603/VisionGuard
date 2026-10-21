"""FindFace-style face tracking and quality scoring."""

from __future__ import annotations

import numpy as np

from computer_vision.identity.quality import assess_face, estimate_pose_degrees
from computer_vision.modules.recognition.face_tracker import FaceTracker

SHARP = np.random.default_rng(3).integers(40, 220, (500, 500, 3), dtype=np.uint8)


def lm(nose_x=200, nose_y=223):
    return np.array([[150, 170], [250, 170], [nose_x, nose_y], [160, 270], [240, 270]], np.float32)


# ---------------------------------------------------------------- tracker ---


def test_a_moving_face_survives_missed_frames_thanks_to_velocity_prediction():
    tracker = FaceTracker(overlap_threshold=0.25)
    ids = []
    for frame in range(5):  # 40 px face walking right at 15 px per frame
        x = 15 * frame
        [tid], _ = tracker.update([(x, 100, x + 40, 140)], frame, frame * 0.04)
        ids.append(tid)
    # The detector misses frames 5 and 6. At frame 7 the face is 45 px further
    # on - no overlap at all with where it was last seen - but it is exactly
    # where its velocity predicts, so it keeps its track.
    tracker.update([], 5, 0.20)
    tracker.update([], 6, 0.24)
    [tid], _ = tracker.update([(105, 100, 145, 140)], 7, 0.28)
    assert set(ids) == {tid}


def test_two_faces_crossing_keep_their_own_tracks():
    tracker = FaceTracker()
    a_ids, b_ids = [], []
    for frame in range(6):
        a = (100 + 10 * frame, 100, 160 + 10 * frame, 160)
        b = (300 - 10 * frame, 100, 360 - 10 * frame, 160)
        (ta, tb), _ = tracker.update([a, b], frame, frame * 0.04)
        a_ids.append(ta)
        b_ids.append(tb)
    assert len(set(a_ids)) == 1 and len(set(b_ids)) == 1 and a_ids[0] != b_ids[0]


def test_track_ends_after_the_miss_interval_in_seconds():
    tracker = FaceTracker(miss_interval=1.0)
    [first], _ = tracker.update([(0, 0, 50, 50)], 0, 0.0)
    _, closed = tracker.update([], 10, 0.9)
    assert closed == []
    _, closed = tracker.update([], 20, 1.2)
    assert closed == [first]
    [second], _ = tracker.update([(0, 0, 50, 50)], 21, 1.3)
    assert second != first


# ---------------------------------------------------------------- quality ---


def test_pose_angles_in_degrees():
    assert estimate_pose_degrees(lm()) [0] == 0.0
    yaw, _ = estimate_pose_degrees(lm(nose_x=230))
    assert 30 < yaw < 40
    assert estimate_pose_degrees(lm(nose_x=260))[0] == 90.0  # full profile
    assert estimate_pose_degrees(np.zeros((5, 2), np.float32)) == (0.0, 0.0)


def test_one_bad_axis_drags_the_whole_score_down():
    good = assess_face(SHARP, (100, 100, 220, 240), lm(), 0.95)
    turned = assess_face(SHARP, (100, 100, 220, 240), lm(nose_x=240), 0.95)
    flat = np.full_like(SHARP, 128)
    blurred = assess_face(flat, (100, 100, 220, 240), lm(), 0.95)
    assert good.score > 0.9
    assert turned.score < good.score - 0.2 and "turned" in turned.problems
    assert blurred.score < 0.2


def test_pitch_is_gated_as_well_as_yaw():
    chin_down = assess_face(SHARP, (100, 100, 220, 240), lm(nose_y=262), 0.95, max_angle=45)
    assert "turned" in chin_down.problems


# ------------------------------------------------------------ motion gate ---

from computer_vision.core.context import FrameContext  # noqa: E402
from computer_vision.inference.face_base import RawFace  # noqa: E402
from computer_vision.modules.recognition.face_detection import FaceDetectionModule  # noqa: E402
from computer_vision.utils.motion import MotionGate  # noqa: E402


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [RawFace(0.9, (100.0, 100.0, 200.0, 220.0), lm())]


def detector_module(**params):
    module = FaceDetectionModule({"max_reuse_frames": 3, **params})
    module._backend = CountingDetector()
    return module


def scene(x: int) -> np.ndarray:
    """A dark room with one bright 'person' block at column ``x``."""
    img = np.full((480, 640, 3), 30, np.uint8)
    img[100:400, x:x + 120] = 220
    return img


def ctx(frame, frame_id):
    return FrameContext(frame=frame, frame_id=frame_id, source_id="t", monotonic_ts=float(frame_id))


def test_still_frames_reuse_the_last_faces_instead_of_rerunning_the_detector():
    module = detector_module()
    for frame_id in range(4):
        c = ctx(SHARP, frame_id)
        module.process(c)
        assert len(c.get_detections("face_detection")) == 1  # faces still published
    assert module._backend.calls == 1


def test_reuse_is_capped_so_the_detector_still_runs_periodically():
    module = detector_module()
    for frame_id in range(9):
        module.process(ctx(SHARP, frame_id))
    assert module._backend.calls == 3  # frames 0, 4 and 8


def test_motion_wakes_the_detector():
    module = detector_module()
    module.process(ctx(scene(100), 0))
    module.process(ctx(scene(160), 1))
    assert module._backend.calls == 2


def test_gate_can_be_turned_off():
    module = detector_module(motion_threshold=0)
    for frame_id in range(3):
        module.process(ctx(SHARP, frame_id))
    assert module._backend.calls == 3


def test_motion_gate_measures_changed_pixels():
    gate = MotionGate(threshold=0.002)
    assert gate(scene(100)) is True      # first frame: nothing to compare with
    assert gate(scene(100)) is False     # identical
    assert gate(scene(130)) is True
    assert gate.last_intensity > 0.002


# ------------------------------------------------------- batched inference ---

import threading  # noqa: E402

from computer_vision.inference.batching import BatchCombiner  # noqa: E402


def test_concurrent_callers_are_batched_and_each_gets_its_own_result():
    calls = []
    gate = threading.Event()

    def run(frames):
        calls.append(len(frames))
        if len(calls) == 1:
            gate.wait(1.0)  # hold the first batch so the others queue up
        return [int(f[0, 0, 0]) for f in frames]

    combiner = BatchCombiner(run, max_batch=8)
    results = {}

    def camera(k):
        results[k] = combiner.submit(np.full((2, 2, 3), k, np.uint8))

    first = threading.Thread(target=camera, args=(0,))
    first.start()
    while not calls:
        pass
    others = [threading.Thread(target=camera, args=(k,)) for k in range(1, 5)]
    for t in others:
        t.start()
    import time as _t
    _t.sleep(0.1)
    gate.set()
    for t in [first, *others]:
        t.join()
    assert results == {k: k for k in range(5)}
    assert calls[0] == 1 and sum(calls) == 5 and len(calls) <= 3  # the waiting four ran together


def test_a_failing_batch_raises_in_every_caller():
    def boom(frames):
        raise ValueError("gpu fell over")

    combiner = BatchCombiner(boom)
    import pytest as _pytest

    with _pytest.raises(ValueError, match="gpu fell over"):
        combiner.submit(np.zeros((2, 2, 3), np.uint8))
