"""Behaviour, scene and identification modules added on top of tracking:

loitering, intrusion_detection, tailgating, abandoned_object,
inventory_monitoring, low_light, weapon_detection, number_plate_recognition
and person_reid.

The rule-based ones are tested on hand-built tracks, because their logic is
geometry and time - no model is involved. The two that wrap networks are
tested with a stand-in for the network; their real-model checks live in
``test_new_models_integration.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from computer_vision.core.config import load_config
from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.core.frame_processor import FrameProcessor
from computer_vision.inference.base import RawDetection
from computer_vision.inference.plate_reader import PlateBox, PlateText, decode_plate, letterbox
from computer_vision.modules.analysis.abandoned_object import AbandonedObjectModule
from computer_vision.modules.analysis.intrusion import (
    IntrusionDetectionModule,
    parse_hours,
    within_hours,
)
from computer_vision.modules.analysis.inventory import InventoryMonitoringModule
from computer_vision.modules.analysis.loitering import LoiteringModule
from computer_vision.modules.analysis.low_light import LowLightModule, enhance_frame, mean_brightness
from computer_vision.modules.analysis.tailgating import TailgatingModule
from computer_vision.modules.base import ModuleInitializationError
from computer_vision.modules.detection.weapon import WeaponDetectionModule
from computer_vision.modules.recognition.number_plate import NumberPlateRecognitionModule
from computer_vision.modules.recognition.person_reid import Gallery, PersonReIdModule
from computer_vision.modules.registry import ModuleRegistry

W, H = 1000, 500


def ctx(t: float = 0.0, frame_id: int = 0, *, frame=None, wall_time=None) -> FrameContext:
    kwargs = {"wall_time": wall_time} if wall_time is not None else {}
    return FrameContext(
        frame=np.zeros((H, W, 3), dtype=np.uint8) if frame is None else frame,
        frame_id=frame_id,
        source_id="cam-a",
        monotonic_ts=t,
        **kwargs,
    )


def person(track_id: int, foot_x: float, foot_y: float, source: str = "person_tracking") -> Track:
    """A 40x100 person whose feet are at (foot_x, foot_y)."""
    return Track(
        track_id=track_id,
        label="person",
        bbox=BBox(foot_x - 20, foot_y - 100, foot_x + 20, foot_y),
        confidence=0.9,
        source_module=source,
    )


def thing(track_id: int, label: str, cx: float, cy: float, size: float = 40) -> Track:
    half = size / 2
    return Track(
        track_id=track_id,
        label=label,
        bbox=BBox(cx - half, cy - half, cx + half, cy + half),
        confidence=0.8,
        source_module="object_tracking",
    )


def build(cls, **params):
    module = cls(params)
    module.initialize()
    return module


def events_of(context: FrameContext, event_type: str):
    return [e for e in context.events if e.event_type == event_type]


def run(module, frames):
    """Feed ``(t, {source: [tracks]}, {source: [detections]})`` tuples; return all events."""
    seen = []
    for frame_id, (t, tracks, detections) in enumerate(frames):
        context = ctx(t, frame_id)
        for source, items in tracks.items():
            context.add_tracks(source, items)
        for source, items in detections.items():
            context.add_detections(source, items)
        module.process(context)
        seen.extend(context.events)
    return seen


# --------------------------------------------------------------------------- #
# Registry and shipped configuration
# --------------------------------------------------------------------------- #

NEW_MODULES = {
    "low_light",
    "loitering",
    "intrusion_detection",
    "tailgating",
    "abandoned_object",
    "inventory_monitoring",
    "weapon_detection",
    "number_plate_recognition",
    "person_reid",
}


class TestRegistration:
    def test_every_new_module_is_known_and_resolves(self):
        assert NEW_MODULES <= set(ModuleRegistry.known_names())
        for name in NEW_MODULES:
            assert ModuleRegistry.resolve(name).module_name == name

    def test_shipped_config_lists_them_all_disabled(self):
        config = load_config("configs/config.yaml", use_env=False)
        assert NEW_MODULES <= set(config.modules)
        assert not (NEW_MODULES & set(config.enabled_module_names()))

    def test_low_light_is_listed_before_every_detector(self):
        # Its brightened frame only helps modules that run after it, and the
        # registry breaks ordering ties by config order.
        order = list(load_config("configs/config.yaml", use_env=False).modules)
        assert order.index("low_light") < order.index("person_detection")

    @pytest.mark.parametrize(
        "name",
        ["loitering", "intrusion_detection", "tailgating", "abandoned_object", "inventory_monitoring"],
    )
    def test_rule_modules_build_from_the_shipped_config(self, name):
        config = load_config("configs/config.yaml", use_env=False)
        module = ModuleRegistry.resolve(name)(config.module_config(name).params)
        module.initialize()


# --------------------------------------------------------------------------- #
# Loitering
# --------------------------------------------------------------------------- #


class TestLoitering:
    def test_standing_still_long_enough_is_reported_once(self):
        module = build(LoiteringModule, seconds=10, cooldown_seconds=0)
        frames = [(t, {"person_tracking": [person(1, 500, 400)]}, {}) for t in range(0, 30)]
        found = [e for e in run(module, frames) if e.event_type == "loitering"]

        assert len(found) == 1
        assert found[0].subject == {"track_id": 1, "class": "person"}
        assert found[0].data["seconds"] >= 10

    def test_walking_across_the_frame_is_not_loitering(self):
        module = build(LoiteringModule, seconds=10)
        frames = [(t, {"person_tracking": [person(1, 50 + 30 * t, 400)]}, {}) for t in range(0, 30)]
        assert run(module, frames) == []

    def test_small_movements_inside_the_radius_still_count(self):
        module = build(LoiteringModule, seconds=10, radius=0.08)
        frames = [
            (t, {"person_tracking": [person(1, 500 + (20 if t % 2 else -20), 400)]}, {})
            for t in range(0, 15)
        ]
        assert len(run(module, frames)) == 1

    def test_leaving_the_spot_restarts_the_clock(self):
        module = build(LoiteringModule, seconds=10)
        frames = [(t, {"person_tracking": [person(1, 200, 400)]}, {}) for t in range(0, 8)]
        frames += [(t, {"person_tracking": [person(1, 800, 400)]}, {}) for t in range(8, 16)]
        assert run(module, frames) == []

    def test_cooldown_repeats_the_event(self):
        module = build(LoiteringModule, seconds=5, cooldown_seconds=10)
        frames = [(t, {"person_tracking": [person(1, 500, 400)]}, {}) for t in range(0, 30)]
        assert len(run(module, frames)) == 3  # at 5s, 15s, 25s

    def test_zones_limit_where_it_applies(self):
        zone = {"name": "door", "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]]}
        module = build(LoiteringModule, seconds=5, zones=[zone])
        outside = [(t, {"person_tracking": [person(1, 800, 400)]}, {}) for t in range(0, 12)]
        inside = [(t, {"person_tracking": [person(2, 100, 400)]}, {}) for t in range(12, 24)]

        found = run(module, outside + inside)
        assert [e.subject["track_id"] for e in found] == [2]
        assert found[0].data["zone"] == "door"

    def test_a_missed_frame_does_not_reset_the_clock(self):
        module = build(LoiteringModule, seconds=10)
        frames = [
            (t, {"person_tracking": [] if t in (4, 5) else [person(1, 500, 400)]}, {})
            for t in range(0, 14)
        ]
        assert len(run(module, frames)) == 1

    def test_bad_configuration_is_a_startup_error(self):
        with pytest.raises(ModuleInitializationError):
            build(LoiteringModule, seconds=0)
        with pytest.raises(ModuleInitializationError):
            build(LoiteringModule, severity="loud")


# --------------------------------------------------------------------------- #
# Intrusion
# --------------------------------------------------------------------------- #


def at_local(hour: int, minute: int = 0) -> datetime:
    """A timezone-aware instant whose *local* clock time is hour:minute."""
    return datetime.now().astimezone().replace(hour=hour, minute=minute, second=0, microsecond=0)


class TestIntrusion:
    def test_hours_parsing_and_wrap_past_midnight(self):
        night = parse_hours("22:00-06:00")
        assert within_hours(night, at_local(23).time())
        assert within_hours(night, at_local(2).time())
        assert not within_hours(night, at_local(12).time())
        assert within_hours(parse_hours(""), at_local(12).time())  # empty = always
        assert within_hours(parse_hours(["09:00-10:00", "13:00-14:00"]), at_local(13, 30).time())

    def test_malformed_hours_are_rejected(self):
        with pytest.raises(ModuleInitializationError):
            build(IntrusionDetectionModule, active_hours="ten till six")

    def _run(self, module, hour, frames=5):
        seen = []
        for i in range(frames):
            context = ctx(float(i), i, wall_time=at_local(hour).astimezone(timezone.utc))
            context.add_tracks("person_tracking", [person(1, 500, 400)])
            module.process(context)
            seen.extend(context.events)
        return seen

    def test_person_during_active_hours_is_an_intrusion(self):
        module = build(IntrusionDetectionModule, active_hours="22:00-06:00", min_frames=3)
        found = self._run(module, hour=23)
        assert [e.event_type for e in found] == ["intrusion"]
        assert found[0].severity.value == "critical"

    def test_same_person_at_midday_is_not(self):
        module = build(IntrusionDetectionModule, active_hours="22:00-06:00", min_frames=3)
        assert self._run(module, hour=12) == []

    def test_one_frame_is_not_enough(self):
        module = build(IntrusionDetectionModule, active_hours="", min_frames=3)
        assert self._run(module, hour=12, frames=2) == []

    def test_zone_restricts_it(self):
        zone = {"name": "vault", "polygon": [[0.0, 0.0], [0.2, 0.0], [0.2, 1.0], [0.0, 1.0]]}
        module = build(IntrusionDetectionModule, active_hours="", zones=[zone])
        assert self._run(module, hour=12) == []  # person stands at x=500, outside


# --------------------------------------------------------------------------- #
# Tailgating
# --------------------------------------------------------------------------- #

# A horizontal line across the middle; feet go from y=200 (above) to y=300.
LINE = [[0.2, 0.5], [0.8, 0.5]]


def walk_through(track_id: int, start: float, x: float = 500, down: bool = True):
    """Three sightings of one person stepping over the line at time ``start``."""
    ys = (200, 240, 300) if down else (300, 260, 200)
    return [(start + 0.2 * i, person(track_id, x, y)) for i, y in enumerate(ys)]


def merge(*walks):
    by_time = {}
    for walk in walks:
        for t, track in walk:
            by_time.setdefault(round(t, 3), []).append(track)
    return [(t, {"person_tracking": tracks}, {}) for t, tracks in sorted(by_time.items())]


class TestTailgating:
    def test_two_people_close_together_is_tailgating(self):
        module = build(TailgatingModule, line=LINE, min_gap_seconds=2.0)
        found = run(module, merge(walk_through(1, 0.0), walk_through(2, 1.0, x=520)))

        assert [e.event_type for e in found] == ["tailgating"]
        assert found[0].subject["track_id"] == 2
        assert found[0].data["follows_track_id"] == 1
        assert found[0].data["gap_seconds"] <= 2.0

    def test_two_people_well_apart_is_not(self):
        module = build(TailgatingModule, line=LINE, min_gap_seconds=2.0)
        assert run(module, merge(walk_through(1, 0.0), walk_through(2, 10.0))) == []

    def test_opposite_directions_is_not(self):
        module = build(TailgatingModule, line=LINE, min_gap_seconds=2.0)
        frames = merge(walk_through(1, 0.0), walk_through(2, 1.0, x=520, down=False))
        assert run(module, frames) == []

    def test_crossings_are_counted_per_direction(self):
        module = build(TailgatingModule, line=LINE)
        run(module, merge(walk_through(1, 0.0), walk_through(2, 10.0), walk_through(3, 20.0, down=False)))
        assert sorted(module.crossings.values()) == [1, 2]

    def test_passing_beside_the_line_does_not_count(self):
        module = build(TailgatingModule, line=LINE)
        run(module, merge(walk_through(1, 0.0, x=950)))  # line ends at x=800
        assert module.crossings == {"in": 0, "out": 0}

    def test_hovering_on_the_line_does_not_count_repeatedly(self):
        module = build(TailgatingModule, line=LINE)
        frames = [(0.1 * i, {"person_tracking": [person(1, 500, 250 + (3 if i % 2 else -3))]}, {}) for i in range(20)]
        run(module, frames)
        assert sum(module.crossings.values()) == 0

    def test_direction_filter(self):
        only_out = build(TailgatingModule, line=LINE, direction="out")
        both = merge(walk_through(1, 0.0), walk_through(2, 1.0, x=520))
        found = run(only_out, both)
        other = build(TailgatingModule, line=LINE, direction="in")
        assert len(found) + len(run(other, both)) == 1  # exactly one of the two fires

    def test_report_crossings_emits_one_event_each(self):
        module = build(TailgatingModule, line=LINE, report_crossings=True)
        found = run(module, merge(walk_through(1, 0.0), walk_through(2, 10.0)))
        assert [e.event_type for e in found] == ["line_crossed", "line_crossed"]

    @pytest.mark.parametrize("line", [None, [[0.1, 0.1]], [[10, 10], [200, 200]], [[0.5, 0.5], [0.5, 0.5]]])
    def test_a_missing_or_invalid_line_is_a_startup_error(self, line):
        with pytest.raises(ModuleInitializationError):
            build(TailgatingModule, line=line)


# --------------------------------------------------------------------------- #
# Abandoned object
# --------------------------------------------------------------------------- #


def someone_at(x: float, y: float) -> Detection:
    return Detection(label="person", confidence=0.9, bbox=BBox(x - 20, y - 100, x + 20, y))


class TestAbandonedObject:
    def test_unattended_still_bag_is_reported_once(self):
        module = build(AbandonedObjectModule, seconds=10)
        frames = [(t, {"object_tracking": [thing(1, "backpack", 500, 300)]}, {"person_detection": []}) for t in range(0, 25)]
        found = run(module, frames)
        assert [e.event_type for e in found] == ["abandoned_object"]
        assert found[0].subject == {"track_id": 1, "class": "backpack"}

    def test_a_bag_with_its_owner_beside_it_is_not(self):
        module = build(AbandonedObjectModule, seconds=10)
        frames = [
            (t, {"object_tracking": [thing(1, "backpack", 500, 300)]}, {"person_detection": [someone_at(520, 330)]})
            for t in range(0, 25)
        ]
        assert run(module, frames) == []

    def test_someone_far_away_does_not_count_as_the_owner(self):
        module = build(AbandonedObjectModule, seconds=10)
        frames = [
            (t, {"object_tracking": [thing(1, "suitcase", 500, 300)]}, {"person_detection": [someone_at(50, 450)]})
            for t in range(0, 25)
        ]
        assert len(run(module, frames)) == 1

    def test_a_passer_by_resets_the_clock(self):
        module = build(AbandonedObjectModule, seconds=10)
        frames = [
            (t, {"object_tracking": [thing(1, "backpack", 500, 300)]},
             {"person_detection": [someone_at(510, 330)] if t == 8 else []})
            for t in range(0, 17)
        ]
        assert run(module, frames) == []  # 8s alone, visit, then only 8s more

    def test_a_carried_bag_is_not(self):
        module = build(AbandonedObjectModule, seconds=10)
        frames = [(t, {"object_tracking": [thing(1, "handbag", 100 + 30 * t, 300)]}, {"person_detection": []}) for t in range(0, 25)]
        assert run(module, frames) == []

    def test_other_classes_are_ignored(self):
        module = build(AbandonedObjectModule, seconds=10)
        frames = [(t, {"object_tracking": [thing(1, "chair", 500, 300)]}, {"person_detection": []}) for t in range(0, 25)]
        assert run(module, frames) == []

    def test_a_new_track_id_for_the_same_bag_keeps_the_clock(self):
        # A passer-by hides the bag; the tracker gives it a new id afterwards.
        module = build(AbandonedObjectModule, seconds=10)
        frames = [(t, {"object_tracking": [thing(1, "backpack", 500, 300)]}, {"person_detection": []}) for t in range(0, 6)]
        frames += [(t, {"object_tracking": []}, {"person_detection": []}) for t in range(6, 8)]
        frames += [(t, {"object_tracking": [thing(2, "backpack", 502, 301)]}, {"person_detection": []}) for t in range(8, 12)]
        found = run(module, frames)
        assert [e.subject["track_id"] for e in found] == [2]

    def test_picking_it_up_afterwards_is_reported(self):
        module = build(AbandonedObjectModule, seconds=5)
        frames = [(t, {"object_tracking": [thing(1, "backpack", 500, 300)]}, {"person_detection": []}) for t in range(0, 8)]
        frames += [(8, {"object_tracking": [thing(1, "backpack", 800, 300)]}, {"person_detection": []})]
        assert [e.event_type for e in run(module, frames)] == ["abandoned_object", "abandoned_object_removed"]


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #


def items(label: str, count: int, x0: float = 100):
    return [
        Detection(label=label, confidence=0.9, bbox=BBox(x0 + 60 * i, 100, x0 + 60 * i + 40, 140))
        for i in range(count)
    ]


class TestInventory:
    def frames(self, counts, label="bottle"):
        return [(float(i), {}, {"object_detection": items(label, n)}) for i, n in enumerate(counts)]

    def test_count_dropping_below_min_is_reported_then_restored(self):
        module = build(InventoryMonitoringModule, items={"bottle": {"min": 3}}, window_frames=5)
        found = run(module, self.frames([3] * 8 + [1] * 8 + [3] * 8))

        assert [e.event_type for e in found] == ["inventory_low", "inventory_restored"]
        assert found[0].data == {"count": 1, "min": 3}

    def test_one_missed_frame_changes_nothing(self):
        module = build(InventoryMonitoringModule, items={"bottle": {"min": 3}}, window_frames=5)
        assert run(module, self.frames([3, 3, 3, 3, 3, 0, 3, 3, 3, 3])) == []

    def test_too_many_is_reported(self):
        module = build(InventoryMonitoringModule, items={"bottle": {"max": 2}}, window_frames=3)
        found = run(module, self.frames([2, 2, 2, 5, 5, 5]))
        assert [e.event_type for e in found] == ["inventory_high"]

    def test_nothing_is_judged_before_the_window_fills(self):
        module = build(InventoryMonitoringModule, items={"bottle": {"min": 3}}, window_frames=10)
        assert run(module, self.frames([0] * 9)) == []

    def test_only_items_inside_the_zone_are_counted(self):
        zone = {"name": "shelf", "polygon": [[0.0, 0.0], [0.2, 0.0], [0.2, 1.0], [0.0, 1.0]]}
        module = build(InventoryMonitoringModule, items={"bottle": {"min": 1}}, window_frames=2, zones=[zone])
        far = [(float(i), {}, {"object_detection": items("bottle", 3, x0=600)}) for i in range(4)]
        assert [e.event_type for e in run(module, far)] == ["inventory_low"]

    @pytest.mark.parametrize(
        "bad", [None, {}, {"bottle": {}}, {"bottle": {"min": -1}}, {"bottle": {"min": 5, "max": 2}}]
    )
    def test_bad_items_are_a_startup_error(self, bad):
        with pytest.raises(ModuleInitializationError):
            build(InventoryMonitoringModule, items=bad)

    def test_module_status_is_not_shadowed(self):
        # `status` is the lifecycle state on every module; the per-item
        # states must not replace it.
        module = build(InventoryMonitoringModule, items={"bottle": {"min": 1}})
        assert module.item_status == {"bottle": "ok"}
        assert hasattr(module.status, "value")


# --------------------------------------------------------------------------- #
# Low light
# --------------------------------------------------------------------------- #


def grey(level: int) -> np.ndarray:
    return np.full((H, W, 3), level, dtype=np.uint8)


class TestLowLight:
    def feed(self, module, levels):
        seen, contexts = [], []
        for i, level in enumerate(levels):
            context = ctx(float(i), i, frame=grey(level))
            module.process(context)
            seen.extend(context.events)
            contexts.append(context)
        return seen, contexts

    def test_brightness_measure(self):
        assert mean_brightness(grey(40)) == pytest.approx(40, abs=1)

    def test_sustained_darkness_then_light_gives_two_events(self):
        module = build(LowLightModule, threshold=50, recover=65, min_frames=3)
        seen, _ = self.feed(module, [120] * 3 + [20] * 5 + [120] * 5)
        assert [e.event_type for e in seen] == ["low_light_started", "low_light_ended"]

    def test_a_brief_dip_is_ignored(self):
        module = build(LowLightModule, threshold=50, min_frames=5)
        seen, _ = self.feed(module, [120, 120, 20, 20, 120, 120, 120])
        assert seen == [] and module.is_dark is False

    def test_hysteresis_keeps_it_dark_between_the_two_thresholds(self):
        module = build(LowLightModule, threshold=50, recover=65, min_frames=2)
        seen, _ = self.feed(module, [20] * 3 + [58] * 6)
        assert [e.event_type for e in seen] == ["low_light_started"]
        assert module.is_dark is True

    def test_enhance_brightens_the_frame_only_while_dark(self):
        module = build(LowLightModule, threshold=50, min_frames=2, enhance=True)
        _, contexts = self.feed(module, [120, 120, 20, 20, 20])
        assert mean_brightness(contexts[1].frame) == pytest.approx(120, abs=2)  # untouched
        assert mean_brightness(contexts[-1].frame) > 60

    def test_enhance_off_never_touches_the_frame(self):
        module = build(LowLightModule, threshold=50, min_frames=2)
        _, contexts = self.feed(module, [20] * 5)
        assert mean_brightness(contexts[-1].frame) == pytest.approx(20, abs=1)

    def test_enhance_frame_keeps_shape_and_type(self):
        frame = np.random.default_rng(0).integers(0, 40, (120, 160, 3), dtype=np.uint8)
        out = enhance_frame(frame)
        assert out.shape == frame.shape and out.dtype == np.uint8
        assert out.mean() > frame.mean()

    def test_recover_below_threshold_is_rejected(self):
        with pytest.raises(ModuleInitializationError):
            build(LowLightModule, threshold=50, recover=40)


# --------------------------------------------------------------------------- #
# Weapon detection
# --------------------------------------------------------------------------- #


def raw(label: str, conf: float = 0.8) -> RawDetection:
    return RawDetection(label=label, confidence=conf, xyxy=(100, 100, 150, 200), class_id=-1)


class TestWeaponDetection:
    def feed(self, module, n, start=0.0, step=1.0):
        seen = []
        for i in range(n):
            context = ctx(start + step * i, i)
            module.process(context)
            seen.extend(context.events)
        return seen

    def test_reports_default_coco_weapon_classes_only(self, fake_detector_backend):
        fake_detector_backend.scripted = [raw("knife"), raw("person"), raw("umbrella")]
        module = build(WeaponDetectionModule, backend="fake")
        context = ctx()
        module.process(context)
        assert [d.label for d in context.get_detections("weapon_detection")] == ["knife"]

    def test_event_after_consecutive_frames_then_cooldown(self, fake_detector_backend):
        fake_detector_backend.scripted = [raw("knife")]
        module = build(WeaponDetectionModule, backend="fake", min_frames=3, cooldown_seconds=30)
        seen = self.feed(module, 10)
        assert [e.event_type for e in seen] == ["weapon_detected"]
        assert seen[0].severity.value == "critical"
        assert seen[0].data["classes"] == ["knife"]

    def test_a_single_frame_is_not_an_event(self, fake_detector_backend):
        fake_detector_backend.scripted_sequence = [[raw("knife")], [], [], []]
        module = build(WeaponDetectionModule, backend="fake", min_frames=3)
        assert self.feed(module, 4) == []

    def test_nothing_seen_nothing_reported(self, fake_detector_backend):
        fake_detector_backend.scripted = [raw("person")]
        module = build(WeaponDetectionModule, backend="fake")
        assert self.feed(module, 6) == []

    def test_custom_labels_for_a_custom_model(self, fake_detector_backend):
        fake_detector_backend.scripted = [raw("pistol"), raw("knife")]
        module = build(WeaponDetectionModule, backend="fake", labels=["pistol"])
        context = ctx()
        module.process(context)
        assert [d.label for d in context.get_detections("weapon_detection")] == ["pistol"]


# --------------------------------------------------------------------------- #
# Number plates
# --------------------------------------------------------------------------- #


class FakePlateReader:
    """Stands in for the two networks: one plate per vehicle, scripted text."""

    def __init__(self, width: int = 120, texts=("MH12AB1234",), confidence: float = 0.95):
        self.width, self.texts, self.confidence = width, list(texts), confidence
        self.detect_calls = 0
        self.read_calls = 0

    def detect(self, image, min_confidence=0.4):
        self.detect_calls += 1
        return [PlateBox(10, 20, 10 + self.width, 50, 0.9)]

    def read(self, plate_image):
        text = self.texts[min(self.read_calls, len(self.texts) - 1)]
        self.read_calls += 1
        return PlateText(text=text, confidence=self.confidence, region="Unknown")


def car(track_id: int = 1) -> Track:
    return Track(track_id=track_id, label="car", bbox=BBox(100, 100, 500, 400),
                 confidence=0.9, source_module="vehicle_tracking")


def anpr(monkeypatch, reader, **params) -> NumberPlateRecognitionModule:
    monkeypatch.setattr(NumberPlateRecognitionModule, "_build_reader", lambda self: reader)
    return build(NumberPlateRecognitionModule, interval_frames=1, **params)


def feed_cars(module, n, original=None):
    seen, last = [], None
    for i in range(n):
        context = ctx(float(i), i)
        context.original_frame = original
        context.add_tracks("vehicle_tracking", [car()])
        module.process(context)
        seen.extend(context.events)
        last = context
    return seen, last


class TestNumberPlate:
    def test_same_text_twice_is_reported_once(self, monkeypatch):
        module = anpr(monkeypatch, FakePlateReader(), votes=2)
        seen, last = feed_cars(module, 6)

        assert [e.event_type for e in seen] == ["plate_read"]
        assert seen[0].data["plate"] == "MH12AB1234"
        assert seen[0].subject == {"track_id": 1, "class": "car"}
        assert [d.label for d in last.get_detections("number_plate_recognition")] == ["MH12AB1234"]

    def test_a_plate_too_small_is_boxed_but_never_read(self, monkeypatch):
        reader = FakePlateReader(width=40)
        module = anpr(monkeypatch, reader, min_plate_width=80)
        seen, last = feed_cars(module, 5)

        assert seen == []
        assert reader.read_calls == 0
        assert [d.label for d in last.get_detections("number_plate_recognition")] == ["plate"]

    def test_a_low_confidence_read_is_discarded(self, monkeypatch):
        module = anpr(monkeypatch, FakePlateReader(confidence=0.5), read_confidence=0.85)
        seen, _ = feed_cars(module, 5)
        assert seen == []

    def test_disagreeing_reads_never_reach_the_vote_count(self, monkeypatch):
        reader = FakePlateReader(texts=["AB12", "AB13", "AB14", "AB15", "AB16"])
        module = anpr(monkeypatch, reader, votes=2)
        seen, _ = feed_cars(module, 5)
        assert seen == []

    def test_plate_box_is_mapped_into_frame_coordinates(self, monkeypatch):
        # Original capture is twice the frame size; the vehicle crop is taken
        # there, and the plate must still land inside the vehicle box here.
        module = anpr(monkeypatch, FakePlateReader(width=120))
        original = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
        _, last = feed_cars(module, 2, original=original)

        plate = last.get_detections("number_plate_recognition")[0].bbox
        vehicle = car().bbox
        assert vehicle.x1 <= plate.x1 < plate.x2 <= vehicle.x2
        assert vehicle.y1 <= plate.y1 < plate.y2 <= vehicle.y2
        # 10px into an 800px-wide crop of a 400px-wide box -> 5px in the frame.
        assert plate.x1 == pytest.approx(vehicle.x1 + 5)

    def test_interval_limits_how_often_a_vehicle_is_examined(self, monkeypatch):
        reader = FakePlateReader()
        monkeypatch.setattr(NumberPlateRecognitionModule, "_build_reader", lambda self: reader)
        module = build(NumberPlateRecognitionModule, interval_frames=5)
        feed_cars(module, 10)
        assert reader.detect_calls == 2

    def test_a_vehicle_too_small_to_carry_a_readable_plate_is_skipped(self, monkeypatch):
        reader = FakePlateReader()
        module = anpr(monkeypatch, reader, min_plate_width=300)  # car is 400px wide
        seen, last = feed_cars(module, 4)
        assert reader.detect_calls == 0
        assert seen == [] and last.get_detections("number_plate_recognition") == []

    def test_decode_plate_strips_padding(self):
        from computer_vision.inference.plate_reader import ALPHABET, PLATE_SLOTS

        probs = np.zeros((PLATE_SLOTS, len(ALPHABET)), dtype=np.float32)
        for slot, char in enumerate("AB123" + "_" * (PLATE_SLOTS - 5)):
            probs[slot, ALPHABET.index(char)] = 0.9
        result = decode_plate(probs)
        assert result.text == "AB123"
        assert result.confidence == pytest.approx(0.9)

    def test_letterbox_pads_to_a_square(self):
        boxed, ratio, (pad_w, pad_h) = letterbox(np.zeros((100, 300, 3), dtype=np.uint8), 384)
        assert boxed.shape == (384, 384, 3)
        assert ratio == pytest.approx(1.28)
        assert pad_w == 0 and pad_h == pytest.approx(128)

    def test_frame_processor_keeps_the_original_capture(self):
        from computer_vision.core.config import ProcessingConfig
        from computer_vision.sources.base import FramePacket

        big = np.zeros((1080, 1920, 3), dtype=np.uint8)
        packet = FramePacket(frame=big, frame_id=1, source_id="x", monotonic_ts=0.0)
        context = FrameProcessor(ProcessingConfig(target_width=960)).build_context(packet)
        assert context.frame.shape[:2] == (540, 960)
        assert context.original_frame is big

        same = FrameProcessor(ProcessingConfig(target_width=1920)).build_context(packet)
        assert same.original_frame is None


# --------------------------------------------------------------------------- #
# Person re-identification
# --------------------------------------------------------------------------- #


def appearance(seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Returns the appearance vector registered for the crop's brightness."""

    def __init__(self):
        self.calls = 0

    def __call__(self, crop):
        self.calls += 1
        return appearance(int(crop[0, 0, 0]))


def reid(monkeypatch, embedder=None, **params) -> PersonReIdModule:
    embedder = embedder or FakeEmbedder()
    monkeypatch.setattr(PersonReIdModule, "_build_embedder", lambda self: embedder)
    return build(PersonReIdModule, samples=2, interval_frames=1, min_height=50, **params)


def frame_with(people):
    """A frame where each (track_id, look, x) person is painted with value ``look``."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    tracks = []
    for track_id, look, x in people:
        frame[300:400, x - 20:x + 20] = look
        tracks.append(person(track_id, x, 400))
    return frame, tracks


def feed_people(module, sequence, source_id="cam-a", start=0):
    seen, last = [], None
    for i, people in enumerate(sequence):
        frame, tracks = frame_with(people)
        context = FrameContext(frame=frame, frame_id=start + i, source_id=source_id, monotonic_ts=float(start + i))
        context.add_tracks("person_tracking", tracks)
        module.process(context)
        seen.extend(context.events)
        last = context
    return seen, last


class TestPersonReId:
    def test_a_returning_person_gets_their_old_identity(self, monkeypatch):
        module = reid(monkeypatch)
        _, first = feed_people(module, [[(1, 50, 300)]] * 3)
        first_id = first.get_tracks("person_tracking")[0].extra["global_id"]

        # Gone long enough to be forgotten by the tracker, back as track 9.
        seen, second = feed_people(module, [[(9, 50, 600)]] * 3, start=100)
        assert [e.event_type for e in seen] == ["person_reidentified"]
        assert seen[0].data["global_id"] == first_id
        assert seen[0].data["other_camera"] is False
        assert second.get_tracks("person_tracking")[0].extra["global_id"] == first_id

    def test_a_different_looking_person_is_a_new_identity(self, monkeypatch):
        module = reid(monkeypatch)
        feed_people(module, [[(1, 50, 300)]] * 3)
        seen, _ = feed_people(module, [[(9, 200, 600)]] * 3, start=100)
        assert seen == []
        assert len(module.gallery.identities) == 2

    def test_two_people_in_view_never_share_an_identity(self, monkeypatch):
        # Identical appearance, both on screen: the second cannot be "the
        # same person" as someone standing next to them.
        module = reid(monkeypatch)
        seen, last = feed_people(module, [[(1, 50, 300), (2, 50, 600)]] * 4)
        ids = [t.extra.get("global_id") for t in last.get_tracks("person_tracking")]
        assert seen == [] and len(set(ids)) == 2

    def test_small_crops_are_skipped(self, monkeypatch):
        embedder = FakeEmbedder()
        monkeypatch.setattr(PersonReIdModule, "_build_embedder", lambda self: embedder)
        module = build(PersonReIdModule, min_height=500)
        feed_people(module, [[(1, 50, 300)]] * 3)
        assert embedder.calls == 0

    def test_max_per_frame_caps_the_work(self, monkeypatch):
        embedder = FakeEmbedder()
        module = reid(monkeypatch, embedder, max_per_frame=1)
        feed_people(module, [[(1, 50, 200), (2, 90, 400), (3, 130, 600)]])
        assert embedder.calls == 1

    def test_another_camera_recognises_the_person_through_the_shared_file(self, monkeypatch, tmp_path):
        shared = str(tmp_path / "gallery.npz")
        gate = reid(monkeypatch, gallery_path=shared)
        _, first = feed_people(gate, [[(1, 50, 300)]] * 3, source_id="gate")
        gate.cleanup()  # writes the gallery

        corridor = reid(monkeypatch, gallery_path=shared)
        seen, _ = feed_people(corridor, [[(4, 50, 500)]] * 3, source_id="corridor")
        assert [e.event_type for e in seen] == ["person_reidentified"]
        assert seen[0].data["previous_source"] == "gate"
        assert seen[0].data["other_camera"] is True
        assert seen[0].data["global_id"] == first.get_tracks("person_tracking")[0].extra["global_id"]

    def test_gallery_round_trips_and_merges_newest(self, tmp_path):
        path = str(tmp_path / "g.npz")
        one = Gallery(path)
        identity = one.enrol(appearance(1), now=100.0, source_id="a")
        one.save()

        two = Gallery(path)
        two.load()
        assert set(two.identities) == {identity.global_id}
        two.identities[identity.global_id].last_seen = 500.0
        two.identities[identity.global_id].source_id = "b"
        two.save()

        one.load()
        assert one.identities[identity.global_id].source_id == "b"

    def test_a_corrupt_gallery_file_is_ignored(self, tmp_path):
        path = tmp_path / "g.npz"
        path.write_bytes(b"not a gallery")
        gallery = Gallery(str(path))
        gallery.load()
        assert gallery.identities == {}

    def test_identities_expire(self):
        gallery = Gallery(ttl_seconds=60)
        gallery.enrol(appearance(1), now=0.0, source_id="a")
        gallery.prune(now=30.0)
        assert len(gallery.identities) == 1
        gallery.prune(now=120.0)
        assert gallery.identities == {}

    def test_threshold_above_one_is_rejected(self, monkeypatch):
        monkeypatch.setattr(PersonReIdModule, "_build_embedder", lambda self: FakeEmbedder())
        with pytest.raises(ModuleInitializationError):
            build(PersonReIdModule, threshold=1.5)
