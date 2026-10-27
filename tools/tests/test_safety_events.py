"""Fire / smoke events, PPE violations, and the startup check that labels exist in the model.

No model files: detections come from the shared fake backend (tools/tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.inference.base import RawDetection
from computer_vision.modules.analysis.ppe_violation import PPEViolationModule
from computer_vision.modules.base import ModuleInitializationError
from computer_vision.modules.detection.fire import FireDetectionModule
from computer_vision.modules.detection.ppe import PPEDetectionModule
from computer_vision.modules.detection.smoke import SmokeDetectionModule
from tools.tests.conftest import FakeDetectorBackend

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def ctx(frame_id: int, t: float) -> FrameContext:
    return FrameContext(frame=FRAME, frame_id=frame_id, source_id="t", monotonic_ts=t)


def detector(cls, **params):
    module = cls({"backend": "fake", "model": "custom.pt", "share_model": False, **params})
    module.initialize()
    return module


# ------------------------------------------------- labels exist in the model ---


def test_a_label_the_model_does_not_have_is_refused_at_startup(monkeypatch):
    monkeypatch.setattr(FakeDetectorBackend, "_names", {0: "hardhat", 1: "safety-vest"}, raising=False)
    with pytest.raises(ModuleInitializationError) as exc:
        detector(PPEDetectionModule, labels=["helmet", "safety-vest"])
    message = str(exc.value)
    assert "['helmet']" in message and "hardhat" in message   # says what is missing and what exists


def test_labels_the_model_has_start_normally(monkeypatch):
    monkeypatch.setattr(FakeDetectorBackend, "_names", {0: "hardhat", 1: "safety-vest"}, raising=False)
    module = detector(PPEDetectionModule, labels=["hardhat", "safety-vest"])
    assert module._backend is not None


def test_a_backend_that_cannot_list_its_classes_is_not_blocked():
    # The fake backend reports no class list: the check is skipped, not failed.
    module = detector(FireDetectionModule, labels=["fire"])
    assert module._backend is not None


# ---------------------------------------------------------------- fire/smoke ---


def run_frames(module, present, start_t=0.0, dt=0.04):
    """Feed frames; ``present[i]`` says whether frame i has a detection. Returns events per frame."""
    out = []
    for i, has in enumerate(present):
        FakeDetectorBackend.scripted = (
            [RawDetection(module._labels[0], 0.8, (100.0, 100.0, 200.0, 220.0))] if has else []
        )
        c = ctx(i, start_t + i * dt)
        module.process(c)
        out.append([e.event_type for e in c.events])
    return out


def test_one_flash_of_fire_raises_nothing():
    fire = detector(FireDetectionModule, min_frames=10, window_frames=15)
    events = run_frames(fire, [True] + [False] * 20)
    assert not any(events)


def test_fire_seen_in_most_recent_frames_raises_one_event_despite_gaps():
    fire = detector(FireDetectionModule, min_frames=10, window_frames=15, cooldown_seconds=60)
    pattern = [True, True, False] * 10            # detected 2 frames in 3 - a real flame flickers
    events = run_frames(fire, pattern)
    flat = [e for frame in events for e in frame]
    assert flat == ["fire_detected"]              # once, then the cooldown holds


def test_the_event_says_where_and_how_sure():
    fire = detector(FireDetectionModule, min_frames=2, window_frames=2)
    FakeDetectorBackend.scripted = [RawDetection("fire", 0.9, (64.0, 48.0, 128.0, 96.0))]
    fire.process(ctx(0, 0.0))
    c = ctx(1, 0.04)
    fire.process(c)
    (event,) = c.events
    assert event.severity.value == "critical"
    assert event.data["confidence"] == 0.9
    assert event.data["bbox"] == [0.1, 0.1, 0.2, 0.2]


def test_smoke_has_its_own_event_and_milder_severity():
    smoke = detector(SmokeDetectionModule, min_frames=3, window_frames=3)
    events = run_frames(smoke, [True] * 3)
    assert events[-1] == ["smoke_detected"]
    assert smoke._alert.severity.value == "warning"


def test_window_smaller_than_min_frames_is_a_config_error():
    with pytest.raises(ModuleInitializationError, match="window_frames"):
        detector(FireDetectionModule, min_frames=10, window_frames=5)


# ------------------------------------------------------------- PPE violation ---


PERSON = (200.0, 100.0, 300.0, 400.0)          # 300 px tall
HEAD = (230.0, 100.0, 270.0, 140.0)            # top of the body
TORSO = (210.0, 180.0, 290.0, 280.0)
FEET = (220.0, 370.0, 280.0, 400.0)


def ppe_frame(i, items, person=PERSON, track_id=1, t=None):
    c = ctx(i, i * 0.04 if t is None else t)
    c.add_tracks("person_tracking", [Track(track_id, "person", BBox(*person), source_module="person_tracking")])
    c.add_detections(
        "ppe_detection",
        [Detection(label, 0.9, BBox(*box), source_module="ppe_detection") for label, box in items],
    )
    return c


def ppe_module(**params):
    module = PPEViolationModule({"required": ["helmet", "vest"], "min_frames": 5, "window_frames": 8, **params})
    module.initialize()
    return module


def feed(module, frames):
    events = []
    for c in frames:
        module.process(c)
        events += c.events
    return events


def test_a_person_wearing_everything_is_never_reported():
    module = ppe_module()
    events = feed(module, [ppe_frame(i, [("helmet", HEAD), ("vest", TORSO)]) for i in range(20)])
    assert events == []


def test_missing_helmet_for_most_frames_is_reported_once_with_what_is_missing():
    module = ppe_module(cooldown_seconds=120)
    events = feed(module, [ppe_frame(i, [("vest", TORSO)]) for i in range(20)])
    assert [e.event_type for e in events] == ["ppe_violation"]
    assert events[0].data["missing"] == ["helmet"]
    assert events[0].subject["track_id"] == 1


def test_a_couple_of_frames_where_the_helmet_is_missed_are_not_a_violation():
    module = ppe_module()
    frames = []
    for i in range(20):
        items = [("vest", TORSO)] + ([] if i % 5 == 0 else [("helmet", HEAD)])  # missed 1 frame in 5
        frames.append(ppe_frame(i, items))
    assert feed(module, frames) == []


def test_a_helmet_held_at_the_feet_does_not_count_as_worn():
    module = ppe_module()
    events = feed(module, [ppe_frame(i, [("helmet", FEET), ("vest", TORSO)]) for i in range(20)])
    assert events and events[0].data["missing"] == ["helmet"]


def test_a_helmet_on_someone_else_does_not_count():
    module = ppe_module()
    other_head = (500.0, 100.0, 540.0, 140.0)
    events = feed(module, [ppe_frame(i, [("helmet", other_head), ("vest", TORSO)]) for i in range(20)])
    assert events and events[0].data["missing"] == ["helmet"]


def test_a_no_helmet_class_is_a_violation_by_itself():
    module = ppe_module(required=[], violation_labels=["no-helmet"])
    events = feed(module, [ppe_frame(i, [("no-helmet", HEAD)]) for i in range(20)])
    assert events and events[0].data["violation_classes"] == ["no-helmet"]


def test_people_too_small_to_judge_are_skipped():
    module = ppe_module(min_person_height=80)
    tiny = (200.0, 100.0, 220.0, 150.0)  # 50 px tall
    assert feed(module, [ppe_frame(i, [], person=tiny) for i in range(20)]) == []


def test_ppe_violation_needs_a_rule():
    with pytest.raises(ModuleInitializationError, match="required"):
        PPEViolationModule({"required": [], "violation_labels": []}).initialize()


def test_ppe_violation_requires_its_detector_and_tracker():
    assert PPEViolationModule.required_modules({}) == ("person_tracking", "ppe_detection")


# ------------------------------------------------------------ weapon events ---

from computer_vision.modules.detection.weapon import WeaponDetectionModule  # noqa: E402

KNIFE_IN_HAND = (250.0, 250.0, 270.0, 290.0)   # inside PERSON's box


def weapon_module(tmp_path=None, **params):
    base = {"backend": "fake", "model": "x.pt", "share_model": False, "min_frames": 3, "window_frames": 5,
            "cooldown_seconds": 30, "save_snapshots": tmp_path is not None}
    if tmp_path is not None:
        base["snapshot_dir"] = str(tmp_path)
    module = WeaponDetectionModule({**base, **params})
    module.initialize()
    return module


def weapon_frames(module, present, persons=()):
    events = []
    for i, has in enumerate(present):
        FakeDetectorBackend.scripted = [RawDetection("knife", 0.7, KNIFE_IN_HAND)] if has else []
        c = ctx(i, i * 0.04)
        if persons:
            c.add_tracks("person_tracking", list(persons))
        module.process(c)
        events += c.events
    return events


def test_a_knife_missed_in_one_frame_still_raises_the_event():
    # Old rule (3 in a row) restarted at every missed frame and never fired here.
    events = weapon_frames(weapon_module(), [True, True, False, True, False, True])
    assert [e.event_type for e in events] == ["weapon_detected"]
    assert events[0].data["frames_seen"] >= 3


def test_the_weapon_is_attributed_to_the_person_holding_it():
    holder = Track(4, "person", BBox(*PERSON), extra={"identity": "Asha"})
    bystander = Track(9, "person", BBox(500.0, 100.0, 600.0, 400.0))
    events = weapon_frames(weapon_module(), [True] * 3, persons=[holder, bystander])
    (event,) = [e for e in events if e.event_type == "weapon_detected"]
    assert event.subject["track_id"] == 4 and event.subject["identity"] == "Asha"
    assert event.data["person_distance"] == 0.0


def test_a_weapon_with_nobody_near_is_reported_without_a_person():
    far = Track(9, "person", BBox(500.0, 100.0, 600.0, 200.0))   # 100 px tall, ~250 px away
    (event,) = weapon_frames(weapon_module(), [True] * 3, persons=[far])
    assert "track_id" not in event.subject and event.data["person"] == "none near"


def test_each_weapon_event_saves_a_crop_and_an_annotated_frame(tmp_path):
    holder = Track(4, "person", BBox(*PERSON))
    events = weapon_frames(weapon_module(tmp_path), [True] * 3, persons=[holder])
    (event,) = [e for e in events if e.event_type == "weapon_detected"]
    for key in ("snapshot", "frame_snapshot"):
        assert (tmp_path / event.data[key]).is_file()
    assert event.data["snapshot"].startswith("weapon_")


def test_shipped_config_starts_weapon_detection_on_the_shared_model_and_objects_skip_weapons():
    from computer_vision.core.config import load_config
    from computer_vision.inference.shared import share_key

    modules = load_config().modules
    weapon, person = dict(modules["weapon_detection"].params), dict(modules["person_detection"].params)
    assert share_key("ultralytics", weapon) == share_key("ultralytics", person)
    excluded = set(modules["object_detection"].params["exclude_labels"])
    assert {"knife", "scissors", "baseball bat"} <= excluded


# ------------------------------------------------- extra weapon models (e.g. guns) ---

from computer_vision.modules.detection.weapon import merge_detections  # noqa: E402

GUN_SPEC = {"model": "firearm.pt", "backend": "fake", "labels": ["0"], "label_map": {"0": "firearm"},
            "confidence_threshold": 0.8}


def test_an_extra_model_adds_its_classes_under_their_published_name():
    FakeDetectorBackend.scripted = [RawDetection("0", 0.9, (300.0, 200.0, 340.0, 230.0)),
                                    RawDetection("knife", 0.7, KNIFE_IN_HAND)]
    module = weapon_module(labels=["knife"], models=[GUN_SPEC])
    c = ctx(0, 0.0)
    module.process(c)
    assert sorted(d.label for d in c.get_detections("weapon_detection")) == ["firearm", "knife"]


def test_an_extra_model_below_its_own_threshold_is_ignored():
    FakeDetectorBackend.scripted = [RawDetection("0", 0.6, (300.0, 200.0, 340.0, 230.0))]
    module = weapon_module(labels=["knife"], models=[GUN_SPEC])   # needs 0.8
    c = ctx(0, 0.0)
    module.process(c)
    assert c.get_detections("weapon_detection") == []


def test_firearm_events_use_the_published_name():
    FakeDetectorBackend.scripted = [RawDetection("0", 0.9, KNIFE_IN_HAND)]
    module = weapon_module(labels=["knife"], models=[GUN_SPEC])
    events = []
    for i in range(3):
        c = ctx(i, i * 0.04); module.process(c); events += c.events
    assert events and events[0].subject["class"] == "firearm" and events[0].data["classes"] == ["firearm"]


def test_an_extra_model_without_the_named_class_is_refused(monkeypatch):
    monkeypatch.setattr(FakeDetectorBackend, "_names", {0: "knife"}, raising=False)
    with pytest.raises(ModuleInitializationError, match="no class named"):
        weapon_module(labels=["knife"], models=[dict(GUN_SPEC, labels=["gun"])])


def test_two_models_seeing_the_same_object_give_one_box():
    a = Detection("knife", 0.6, BBox(100, 100, 150, 200))
    b = Detection("knife", 0.8, BBox(102, 101, 151, 199))
    other = Detection("firearm", 0.9, BBox(300, 100, 340, 130))
    merged = merge_detections([[a], [b, other]])
    assert sorted((d.label, d.confidence) for d in merged) == [("firearm", 0.9), ("knife", 0.8)]


# ------------------------------------------------------- armed-person tracking ---

from computer_vision.modules.detection.weapon import ARMED_PERSONS_KEY  # noqa: E402

PERSON_B = (400.0, 100.0, 500.0, 400.0)
KNIFE_B = (450.0, 250.0, 470.0, 290.0)    # inside PERSON_B's box


def armed_run(module, steps):
    """``steps``: ``(t, knife_boxes, persons)``; returns ``[(context, events)]`` per frame."""
    out = []
    for i, (t, knives, persons) in enumerate(steps):
        FakeDetectorBackend.scripted = [RawDetection("knife", 0.7, box) for box in knives]
        c = ctx(i, t)
        c.add_tracks("person_tracking", list(persons))
        module.process(c)
        out.append((c, [e for e in c.events if e.event_type == "armed_person"]))
    return out


def armed_events(frames):
    return [e for _, events in frames for e in events]


def test_a_person_holding_a_weapon_is_marked_armed_once():
    holder = Track(4, "person", BBox(*PERSON))
    frames = armed_run(weapon_module(), [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(5)])
    (event,) = armed_events(frames)
    assert event.subject == {"track_id": 4, "class": "knife"}
    assert event.severity.name.lower() == "critical"
    assert holder.extra["armed"] is True and holder.extra["weapon"] == "knife"
    last, _ = frames[-1]
    assert last.metadata[ARMED_PERSONS_KEY][4]["weapon"] == "knife"


def test_one_flash_of_a_weapon_does_not_arm_anyone():
    holder = Track(4, "person", BBox(*PERSON))
    steps = [(i * 0.04, [KNIFE_IN_HAND] if i == 1 else [], [holder]) for i in range(6)]
    assert armed_events(armed_run(weapon_module(), steps)) == []
    assert "armed" not in holder.extra


def test_the_mark_stays_while_the_weapon_is_hidden():
    holder = Track(4, "person", BBox(*PERSON))
    steps = [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(3)]
    steps += [(1.0 + i, [], [holder]) for i in range(30)]           # 30 s, knife out of sight
    frames = armed_run(weapon_module(armed_hold_seconds=60), steps)
    assert len(armed_events(frames)) == 1
    last, _ = frames[-1]
    assert holder.extra["armed"] is True
    assert last.metadata[ARMED_PERSONS_KEY][4]["weapon_visible"] is False


def test_the_mark_is_dropped_after_the_hold_time_without_a_weapon():
    holder = Track(4, "person", BBox(*PERSON))
    steps = [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(3)]
    steps += [(1.0 + i, [], [holder]) for i in range(15)]
    frames = armed_run(weapon_module(armed_hold_seconds=10), steps)
    assert "armed" not in holder.extra
    last, _ = frames[-1]
    assert ARMED_PERSONS_KEY not in last.metadata


def test_one_armed_event_per_person_however_long_they_hold_it():
    holder = Track(4, "person", BBox(*PERSON))
    steps = [(i * 0.5, [KNIFE_IN_HAND], [holder]) for i in range(120)]  # 60 s
    frames = armed_run(weapon_module(cooldown_seconds=1), steps)
    assert len(armed_events(frames)) == 1


def test_each_armed_person_gets_their_own_event_and_bystanders_none():
    a = Track(4, "person", BBox(*PERSON))
    b = Track(7, "person", BBox(*PERSON_B))
    bystander = Track(9, "person", BBox(560.0, 100.0, 630.0, 400.0))
    steps = [(i * 0.04, [KNIFE_IN_HAND, KNIFE_B], [a, b, bystander]) for i in range(4)]
    events = armed_events(armed_run(weapon_module(), steps))
    assert sorted(e.subject["track_id"] for e in events) == [4, 7]
    assert "armed" not in bystander.extra


def test_an_armed_person_recognised_later_is_named_in_one_more_event():
    holder = Track(4, "person", BBox(*PERSON))
    module = weapon_module()
    first = armed_events(armed_run(module, [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(3)]))
    assert len(first) == 1 and "identity" not in first[0].subject

    holder.extra["identity"] = "Asha"     # face_recognition named them
    later = armed_events(armed_run(module, [(1.0 + i, [], [holder]) for i in range(5)]))
    (event,) = later
    assert event.subject["identity"] == "Asha" and event.data["identified_later"] is True


def test_a_person_recognised_from_the_start_is_named_in_the_first_event_only():
    holder = Track(4, "person", BBox(*PERSON), extra={"identity": "Asha"})
    (event,) = armed_events(armed_run(weapon_module(), [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(8)]))
    assert event.subject["identity"] == "Asha" and "identified_later" not in event.data


def test_a_person_who_left_is_forgotten():
    holder = Track(4, "person", BBox(*PERSON))
    module = weapon_module(armed_forget_seconds=5)
    armed_run(module, [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(3)])
    armed_run(module, [(10.0, [], [])])                 # nobody in view for 10 s
    holder.extra.pop("armed", None)                     # a new track object would start clean
    frames = armed_run(module, [(10.5, [], [holder])])
    assert "armed" not in holder.extra and armed_events(frames) == []


def test_armed_tracking_can_be_switched_off():
    holder = Track(4, "person", BBox(*PERSON))
    frames = armed_run(weapon_module(track_armed=False), [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(5)])
    assert armed_events(frames) == [] and "armed" not in holder.extra


def test_the_armed_event_saves_a_picture_of_the_person(tmp_path):
    holder = Track(4, "person", BBox(*PERSON))
    (event,) = armed_events(armed_run(weapon_module(tmp_path), [(i * 0.04, [KNIFE_IN_HAND], [holder]) for i in range(3)]))
    assert (tmp_path / event.data["snapshot"]).is_file()


def test_shipped_config_turns_armed_tracking_on():
    from computer_vision.core.config import load_config

    params = load_config().modules["weapon_detection"].params
    assert params["track_armed"] is True and params["armed_hold_seconds"] > 0


def test_an_armed_person_is_drawn_red_with_the_weapon(monkeypatch):
    from computer_vision.utils import annotate

    captions = []
    real = annotate.draw_box
    monkeypatch.setattr(annotate, "draw_box", lambda f, b, caption="", color=annotate.COLOR_OK, thickness=2:
                        (captions.append((caption, color, thickness)), real(f, b, caption, color, thickness)))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    armed = Track(4, "person", BBox(*PERSON), extra={"armed": True, "weapon": "firearm", "identity": "Asha"})
    calm = Track(7, "person", BBox(*PERSON_B))
    annotate.draw_tracks(frame, [armed, calm])
    (armed_caption, armed_color, armed_thickness), (calm_caption, calm_color, _) = captions
    assert armed_caption == "#4 Asha - ARMED (firearm)"
    assert armed_color == annotate.COLOR_ARMED and armed_thickness == 3
    assert "ARMED" not in calm_caption and calm_color != annotate.COLOR_ARMED
