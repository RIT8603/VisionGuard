"""Pose detection, fall detection, attendance, and the accuracy harness.

Fed scripted poses and recognitions, so no model is needed. The fall geometry
is tested against postures built by hand, which is the only way to assert that
"lying down" scores higher than "standing".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import build_config
from app.core.context import BBox, Detection, FrameContext, Keypoint, Pose, Severity
from app.models.pose_backend import PoseBackend, RawPose, create_pose_backend
from app.models import pose_backend as pose_backend_module
from app.modules.analysis.attendance import (
    EVENT_CHECK_IN,
    EVENT_CHECK_OUT,
    AttendanceModule,
)
from app.modules.analysis.fall import (
    EVENT_FALL,
    EVENT_RECOVERED,
    FallDetectionModule,
    fall_score,
)
from app.modules.base import ModuleInitializationError, ModuleType
from app.modules.detection.pose import PoseDetectionModule
from app.modules.registry import ModuleDependencyError, ModuleRegistry


class FakePoseBackend(PoseBackend):
    name = "fake"
    scripted: list = []
    fail_on_load = False

    def load(self):
        if FakePoseBackend.fail_on_load:
            from app.models.base import ModelLoadError

            raise ModelLoadError("no pose model")
        self._loaded = True

    def estimate(self, frame):
        return list(FakePoseBackend.scripted)


@pytest.fixture(autouse=True)
def fake_pose(monkeypatch):
    FakePoseBackend.scripted = []
    FakePoseBackend.fail_on_load = False
    monkeypatch.setitem(
        pose_backend_module.POSE_BACKENDS,
        "fake",
        "tests.test_pose_fall_attendance:FakePoseBackend",
    )
    yield


def ctx(frame_id: int = 1, ts: float = 0.0) -> FrameContext:
    return FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=ts,
    )


# --------------------------------------------------------------------------- #
# Posture builders
# --------------------------------------------------------------------------- #

def _pose(bbox, joints: dict, confidence: float = 0.9) -> Pose:
    return Pose(
        bbox=BBox(*bbox),
        keypoints=[Keypoint(name, x, y, 0.9) for name, (x, y) in joints.items()],
        confidence=confidence,
    )


def standing_pose(x: float = 100) -> Pose:
    """Tall box, vertical torso, full height."""
    return _pose(
        (x, 100, x + 60, 400),
        {
            "nose": (x + 30, 110),
            "left_shoulder": (x + 15, 150), "right_shoulder": (x + 45, 150),
            "left_hip": (x + 18, 260), "right_hip": (x + 42, 260),
            "left_ankle": (x + 20, 395), "right_ankle": (x + 40, 395),
        },
    )


def lying_pose(y: float = 350) -> Pose:
    """Wide box, horizontal torso, collapsed height."""
    return _pose(
        (100, y, 400, y + 70),
        {
            "nose": (110, y + 30),
            "left_shoulder": (150, y + 25), "right_shoulder": (150, y + 45),
            "left_hip": (280, y + 25), "right_hip": (280, y + 45),
            "left_ankle": (390, y + 30), "right_ankle": (390, y + 50),
        },
    )


# --------------------------------------------------------------------------- #
# Fall geometry
# --------------------------------------------------------------------------- #

class TestFallScore:
    def test_standing_scores_low(self):
        score, signals = fall_score(standing_pose())
        assert score < 0.3
        assert set(signals) == {"box_ratio", "torso_angle", "height_collapse"}

    def test_lying_scores_high(self):
        score, _ = fall_score(lying_pose())
        assert score > 0.8

    def test_lying_beats_standing(self):
        assert fall_score(lying_pose())[0] > fall_score(standing_pose())[0]

    def test_works_with_only_a_box(self):
        """A person whose joints are all occluded still gets an answer."""
        bare = Pose(bbox=BBox(100, 300, 400, 380), keypoints=[], confidence=0.9)

        score, signals = fall_score(bare)

        assert list(signals) == ["box_ratio"]
        assert score > 0.8  # wide box alone is suggestive

    def test_no_geometry_at_all_scores_zero(self):
        empty = Pose(bbox=BBox(10, 10, 10, 10), keypoints=[], confidence=0.9)
        score, signals = fall_score(empty)
        assert score == 0.0 and signals == {}

    def test_low_confidence_joints_are_ignored(self):
        """An occluded joint must not be treated as a measurement."""
        pose = Pose(
            bbox=BBox(100, 100, 160, 400),
            keypoints=[
                Keypoint("left_shoulder", 115, 150, 0.05),
                Keypoint("left_hip", 118, 260, 0.05),
            ],
            confidence=0.9,
        )
        _, signals = fall_score(pose)
        assert "torso_angle" not in signals

    def test_score_is_bounded(self):
        for pose in (standing_pose(), lying_pose()):
            score, _ = fall_score(pose)
            assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Pose detection module
# --------------------------------------------------------------------------- #

def build_pose(**params) -> PoseDetectionModule:
    params.setdefault("backend", "fake")
    module = PoseDetectionModule(params)
    module.initialize()
    return module


def raw_pose(x=100, conf=0.9, joint_conf=0.9) -> RawPose:
    joints = np.zeros((17, 3), dtype=np.float32)
    for index in range(17):
        joints[index] = (x + index, 100 + index * 15, joint_conf)
    return RawPose(confidence=conf, xyxy=(x, 100, x + 60, 400), keypoints=joints)


class TestPoseDetection:
    def test_is_a_detection_module(self):
        assert PoseDetectionModule.module_type is ModuleType.DETECTION
        assert PoseDetectionModule.requires == ()

    def test_publishes_poses(self):
        FakePoseBackend.scripted = [raw_pose(100), raw_pose(300)]
        module = build_pose()
        context = ctx()

        result = module.process(context)

        assert len(context.get_poses("pose_detection")) == 2
        assert result.data["count"] == 2

    def test_also_publishes_person_boxes_for_trackers(self):
        """So a tracker can consume pose_detection without knowing about joints."""
        FakePoseBackend.scripted = [raw_pose(100)]
        module = build_pose()
        context = ctx()

        module.process(context)

        detections = context.get_detections("pose_detection")
        assert [d.label for d in detections] == ["person"]

    def test_keypoints_are_named(self):
        FakePoseBackend.scripted = [raw_pose(100)]
        module = build_pose()
        context = ctx()

        module.process(context)

        pose = context.get_poses("pose_detection")[0]
        assert pose.keypoint("nose") is not None
        assert pose.keypoint("left_ankle") is not None
        assert len(pose.keypoints) == 17

    def test_low_confidence_poses_are_dropped(self):
        FakePoseBackend.scripted = [raw_pose(conf=0.9), raw_pose(300, conf=0.1)]
        module = build_pose(confidence_threshold=0.5)
        context = ctx()

        module.process(context)

        assert len(context.get_poses("pose_detection")) == 1

    def test_weak_keypoints_become_invisible(self):
        FakePoseBackend.scripted = [raw_pose(joint_conf=0.1)]
        module = build_pose(keypoint_threshold=0.5)
        context = ctx()

        module.process(context)

        pose = context.get_poses("pose_detection")[0]
        assert all(not k.is_visible for k in pose.keypoints)

    def test_result_is_json_safe(self):
        import json

        FakePoseBackend.scripted = [raw_pose()]
        module = build_pose()

        result = module.process(ctx())

        json.dumps(result.to_dict())

    def test_summary_includes_poses(self):
        import json

        FakePoseBackend.scripted = [raw_pose()]
        module = build_pose()
        context = ctx()
        module.process(context)

        summary = context.summary()
        json.dumps(summary)
        assert "pose_detection" in summary["poses"]

    def test_backend_failure_is_a_module_error(self):
        FakePoseBackend.fail_on_load = True
        with pytest.raises(ModuleInitializationError, match="no pose model"):
            build_pose()

    def test_unknown_backend_is_reported(self):
        from app.models.base import BackendError

        with pytest.raises(BackendError, match="Unknown pose backend"):
            create_pose_backend("telepathy")


# --------------------------------------------------------------------------- #
# Fall detection module
# --------------------------------------------------------------------------- #

def build_fall(**params) -> FallDetectionModule:
    module = FallDetectionModule(params)
    module.initialize()
    return module


def pose_ctx(poses, frame_id: int = 1, ts: float = 0.0) -> FrameContext:
    context = ctx(frame_id, ts)
    context.add_poses("pose_detection", list(poses))
    return context


class TestFallDetection:
    def test_is_analysis_requiring_pose(self):
        assert FallDetectionModule.module_type is ModuleType.ANALYSIS
        assert FallDetectionModule.requires == ("pose_detection",)

    def test_loads_no_model(self):
        module = build_fall()
        assert not hasattr(module, "_backend")

    def test_standing_raises_nothing(self):
        module = build_fall(min_frames=1)

        for frame in range(5):
            context = pose_ctx([standing_pose()], frame, ts=frame * 0.1)
            module.process(context)
            assert context.events == []

    def test_a_sustained_fall_is_reported(self):
        module = build_fall(min_frames=3)

        fired = []
        for frame in range(5):
            context = pose_ctx([lying_pose()], frame, ts=frame * 0.1)
            module.process(context)
            fired.append(len(context.events))

        assert fired == [0, 0, 1, 0, 0]

    def test_a_single_frame_is_not_a_fall(self):
        """Bending to tie a shoelace looks like the start of a fall."""
        module = build_fall(min_frames=5)
        context = pose_ctx([lying_pose()])

        module.process(context)

        assert context.events == []

    def test_standing_up_resets_the_streak(self):
        """Standing changes the box shape enough that association may drop
        the id; either way the fall streak must not survive it."""
        module = build_fall(min_frames=3, report_recovery=False)
        module.process(pose_ctx([lying_pose()], 1, ts=0.0))
        module.process(pose_ctx([lying_pose()], 2, ts=0.1))
        module.process(pose_ctx([standing_pose()], 3, ts=0.2))

        context = pose_ctx([lying_pose()], 4, ts=0.3)
        module.process(context)

        assert [e.event_type for e in context.events] == []

    def test_a_vanished_fallen_person_does_not_stay_fallen(self):
        """Otherwise the fallen count drifts upward for the whole run."""
        module = build_fall(min_frames=1)
        module.process(pose_ctx([lying_pose()], 1, ts=0.0))

        gone = pose_ctx([], 2, ts=1.0)
        result = module.process(gone)

        assert result.data["fallen"] == 0
        assert [e.event_type for e in gone.events] == [EVENT_RECOVERED]
        assert gone.events[0].data["reason"] == "track_lost"

    def test_cooldown_prevents_repeat_alarms(self):
        module = build_fall(min_frames=1, cooldown_seconds=10)

        total = 0
        for frame in range(30):
            context = pose_ctx([lying_pose()], frame, ts=frame * 0.1)
            module.process(context)
            total += len(context.events)

        assert total == 1

    def test_recovery_is_reported(self):
        module = build_fall(min_frames=1, report_recovery=True)
        module.process(pose_ctx([lying_pose()], 1, ts=0.0))

        context = pose_ctx([standing_pose()], 2, ts=1.0)
        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_RECOVERED]
        assert context.events[0].severity is Severity.INFO

    def test_recovery_can_be_silenced(self):
        module = build_fall(min_frames=1, report_recovery=False)
        module.process(pose_ctx([lying_pose()], 1, ts=0.0))

        context = pose_ctx([standing_pose()], 2, ts=1.0)
        module.process(context)

        assert context.events == []

    def test_event_explains_which_signals_fired(self):
        """A false alarm should be diagnosable, not just disbelieved."""
        module = build_fall(min_frames=1)
        context = pose_ctx([lying_pose()])

        module.process(context)

        event = context.events[0]
        assert event.event_type == EVENT_FALL
        assert event.severity is Severity.CRITICAL
        assert "box_ratio" in event.data["signals"]
        assert 0.0 <= event.data["score"] <= 1.0

    def test_two_people_are_scored_independently(self):
        module = build_fall(min_frames=1)
        context = pose_ctx([standing_pose(x=50), lying_pose()])

        result = module.process(context)

        assert result.data["people"] == 2
        assert len(context.events) == 1  # only the fallen one

    def test_no_poses_is_quiet(self):
        module = build_fall(min_frames=1)
        context = pose_ctx([])

        result = module.process(context)

        assert context.events == []
        assert result.data["people"] == 0

    def test_bad_severity_is_rejected(self):
        with pytest.raises(ValueError, match="severity"):
            FallDetectionModule({"severity": "catastrophic"})


# --------------------------------------------------------------------------- #
# Attendance
# --------------------------------------------------------------------------- #

def build_attendance(**params) -> AttendanceModule:
    module = AttendanceModule(params)
    module.initialize()
    return module


def seen_ctx(names, frame_id: int = 1, ts: float = 0.0) -> FrameContext:
    context = ctx(frame_id, ts)
    context.add_detections(
        "face_recognition",
        [
            Detection(
                label=name,
                confidence=0.9,
                bbox=BBox(100 + i * 70, 50, 160 + i * 70, 120),
                extra={"recognised": True, "identity": name, "similarity": 0.8},
            )
            for i, name in enumerate(names)
        ],
    )
    return context


class TestAttendance:
    def test_is_analysis_requiring_recognition(self):
        assert AttendanceModule.module_type is ModuleType.ANALYSIS
        assert AttendanceModule.requires == ("face_recognition",)

    def test_check_in_after_corroboration(self):
        module = build_attendance(min_sightings=3)

        fired = []
        for frame in range(4):
            context = seen_ctx(["alice"], frame, ts=frame * 0.1)
            module.process(context)
            fired.append([e.event_type for e in context.events])

        assert fired == [[], [], [EVENT_CHECK_IN], []]
        assert module.present == ["alice"]

    def test_one_frame_is_not_an_arrival(self):
        """Recognition has a false-positive rate; a wrong check-in is worse
        than a late one."""
        module = build_attendance(min_sightings=3)
        context = seen_ctx(["alice"])

        module.process(context)

        assert context.events == []
        assert module.present == []

    def test_unrecognised_faces_are_ignored(self):
        module = build_attendance(min_sightings=1)
        context = ctx()
        context.add_detections(
            "face_recognition",
            [
                Detection(
                    label="unknown",
                    confidence=0.9,
                    bbox=BBox(10, 10, 60, 60),
                    extra={"recognised": False, "similarity": 0.1},
                )
            ],
        )

        module.process(context)

        assert context.events == []

    def test_check_out_after_absence(self):
        module = build_attendance(min_sightings=1, absence_timeout=5.0)
        module.process(seen_ctx(["alice"], 1, ts=0.0))

        still_there = seen_ctx([], 2, ts=3.0)
        module.process(still_there)
        assert still_there.events == []

        gone = seen_ctx([], 3, ts=10.0)
        module.process(gone)

        assert [e.event_type for e in gone.events] == [EVENT_CHECK_OUT]
        assert module.present == []

    def test_check_out_reports_duration(self):
        module = build_attendance(min_sightings=1, absence_timeout=5.0)
        module.process(seen_ctx(["alice"], 1, ts=0.0))
        module.process(seen_ctx(["alice"], 2, ts=30.0))

        gone = seen_ctx([], 3, ts=40.0)
        module.process(gone)

        assert gone.events[0].data["duration_seconds"] == pytest.approx(30.0, abs=0.1)

    def test_brief_occlusion_does_not_check_out(self):
        """They turned around; they did not leave."""
        module = build_attendance(min_sightings=1, absence_timeout=60.0)
        module.process(seen_ctx(["alice"], 1, ts=0.0))

        for frame in range(2, 10):
            context = seen_ctx([], frame, ts=frame * 2.0)
            module.process(context)
            assert context.events == []

        assert module.present == ["alice"]

    def test_returning_needs_fresh_corroboration(self):
        module = build_attendance(min_sightings=2, absence_timeout=5.0)
        for frame in (1, 2):
            module.process(seen_ctx(["alice"], frame, ts=0.0))
        module.process(seen_ctx([], 3, ts=20.0))  # checked out

        back = seen_ctx(["alice"], 4, ts=21.0)
        module.process(back)
        assert back.events == []  # one sighting is not enough again

        confirmed = seen_ctx(["alice"], 5, ts=21.1)
        module.process(confirmed)
        assert [e.event_type for e in confirmed.events] == [EVENT_CHECK_IN]

    def test_sessions_accumulate(self):
        module = build_attendance(min_sightings=1, absence_timeout=5.0)
        module.process(seen_ctx(["alice"], 1, ts=0.0))
        module.process(seen_ctx(["alice"], 2, ts=10.0))
        module.process(seen_ctx([], 3, ts=20.0))

        record = module.record("alice")
        assert len(record.sessions) == 1
        assert record.total_seconds == pytest.approx(10.0, abs=0.1)

    def test_several_people_tracked_separately(self):
        module = build_attendance(min_sightings=1)
        context = seen_ctx(["alice", "bob"])

        result = module.process(context)

        assert sorted(module.present) == ["alice", "bob"]
        assert result.data["present_count"] == 2

    def test_result_is_json_safe(self):
        import json

        module = build_attendance(min_sightings=1)
        result = module.process(seen_ctx(["alice"]))
        json.dumps(result.to_dict())


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

class TestWiring:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("pose_detection", PoseDetectionModule),
            ("fall_detection", FallDetectionModule),
            ("attendance", AttendanceModule),
        ],
    )
    def test_resolves_by_name(self, name, cls):
        assert ModuleRegistry.resolve(name) is cls

    def test_pose_runs_before_fall(self):
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        "fall_detection": {"enabled": True},
                        "pose_detection": {"enabled": True, "backend": "fake"},
                    }
                }
            )
        )
        assert [m.module_name for m in built] == ["pose_detection", "fall_detection"]

    def test_fall_without_pose_is_rejected(self):
        with pytest.raises(ModuleDependencyError, match="pose_detection"):
            ModuleRegistry.build_enabled(
                build_config({"modules": {"fall_detection": {"enabled": True}}})
            )

    def test_attendance_without_recognition_is_rejected(self):
        with pytest.raises(ModuleDependencyError, match="face_recognition"):
            ModuleRegistry.build_enabled(
                build_config({"modules": {"attendance": {"enabled": True}}})
            )

    def test_new_modules_ship_disabled(self):
        from app.core.config import load_config

        config = load_config("configs/config.yaml", use_env=False)

        for name in ("pose_detection", "fall_detection", "attendance"):
            assert config.modules[name].enabled is False

    def test_full_chain_through_the_pipeline(self):
        from app.core.frame_processor import FrameProcessor
        from app.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink, SyntheticSource

        # A person lying on the floor, for the whole clip.
        joints = np.zeros((17, 3), dtype=np.float32)
        layout = {
            0: (110, 380), 5: (150, 375), 6: (150, 395),
            11: (280, 375), 12: (280, 395), 15: (390, 380), 16: (390, 400),
        }
        for index, (x, y) in layout.items():
            joints[index] = (x, y, 0.9)
        FakePoseBackend.scripted = [
            RawPose(confidence=0.9, xyxy=(100, 350, 400, 420), keypoints=joints)
        ]

        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "pose_detection": {"enabled": True, "backend": "fake"},
                    "fall_detection": {"enabled": True, "min_frames": 2},
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
        assert [e.event_type for e in events] == [EVENT_FALL]


# --------------------------------------------------------------------------- #
# Accuracy harness
# --------------------------------------------------------------------------- #

class TestEvaluationHarness:
    def _store_with(self, **people):
        from app.identity import IdentityStore

        store = IdentityStore(threshold=0.5)
        for name, embedding in people.items():
            store.enroll(name, embedding)
        return store

    def test_scores_a_perfect_run(self):
        from tools.evaluate_recognition import Sample, evaluate

        alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store = self._store_with(alice=alice)
        samples = [Sample("alice", Path("a.jpg"), alice)]

        outcome = evaluate(store, samples, 0.5, {"alice"})

        assert outcome.accuracy == 1.0
        assert outcome.false_accept == 0 and outcome.false_reject == 0

    def test_counts_a_false_reject(self):
        from tools.evaluate_recognition import Sample, evaluate

        alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store = self._store_with(alice=alice)
        samples = [Sample("alice", Path("a.jpg"), alice)]

        outcome = evaluate(store, samples, 0.99999, {"alice"})

        assert outcome.false_reject_rate >= 0.0

    def test_counts_a_false_accept(self):
        from tools.evaluate_recognition import Sample, evaluate

        alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        stranger = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        store = self._store_with(alice=alice)
        samples = [Sample("mallory", Path("m.jpg"), stranger)]

        outcome = evaluate(store, samples, 0.5, {"alice"})

        assert outcome.strangers == 1
        assert outcome.false_accept == 1
        assert outcome.false_accept_rate == 1.0

    def test_counts_a_wrong_identity(self):
        from tools.evaluate_recognition import Sample, evaluate

        alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        store = self._store_with(alice=alice, bob=bob)
        # A "bob" image that actually looks like alice.
        samples = [Sample("bob", Path("b.jpg"), alice)]

        outcome = evaluate(store, samples, 0.5, {"alice", "bob"})

        assert outcome.wrong_identity == 1
        assert outcome.correct == 0

    def test_sweep_covers_every_threshold(self):
        from tools.evaluate_recognition import Sample, sweep

        alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store = self._store_with(alice=alice)
        samples = [Sample("alice", Path("a.jpg"), alice)]

        outcomes = sweep(store, samples, [0.2, 0.4, 0.6], {"alice"})

        assert [o.threshold for o in outcomes] == [0.2, 0.4, 0.6]

    def test_raising_the_threshold_trades_far_for_frr(self):
        """The core reason the sweep exists.

        Enrolled alice, a *different* photo of alice (cos ~0.98), and a
        lookalike stranger (cos ~0.90). No single threshold is free.
        """
        from tools.evaluate_recognition import Sample, evaluate

        enrolled = np.array([1.0, 0.0], dtype=np.float32)
        alice_other_photo = np.array([0.98, 0.199], dtype=np.float32)
        lookalike = np.array([0.90, 0.436], dtype=np.float32)

        store = self._store_with(alice=enrolled)
        samples = [
            Sample("alice", Path("a.jpg"), alice_other_photo),
            Sample("mallory", Path("m.jpg"), lookalike),
        ]

        lenient = evaluate(store, samples, 0.85, {"alice"})
        balanced = evaluate(store, samples, 0.95, {"alice"})
        strict = evaluate(store, samples, 0.995, {"alice"})

        # Too low: the stranger gets in.
        assert lenient.false_accept == 1 and lenient.false_reject == 0
        # Just right for this data.
        assert balanced.false_accept == 0 and balanced.false_reject == 0
        # Too high: alice is turned away.
        assert strict.false_accept == 0 and strict.false_reject == 1

    def test_finds_people_directories(self, tmp_path):
        from tools.evaluate_recognition import find_people

        (tmp_path / "alice").mkdir()
        (tmp_path / "alice" / "a.jpg").write_bytes(b"x")
        (tmp_path / "bob").mkdir()
        (tmp_path / "bob" / "b.png").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("ignored")
        (tmp_path / "empty").mkdir()

        people = find_people(tmp_path)

        assert sorted(people) == ["alice", "bob"]  # 'empty' has no images

    def test_missing_directory_is_reported(self, tmp_path):
        from tools.evaluate_recognition import find_people

        with pytest.raises(SystemExit, match="Not a directory"):
            find_people(tmp_path / "nope")
