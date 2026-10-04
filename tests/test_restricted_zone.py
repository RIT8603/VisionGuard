"""Restricted Zone: geometry, the entry/exit state machine, and the module.

Fed scripted tracks, which is the only way to assert event timing
deterministically. No model, no camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from computer_vision.core.config import build_config
from computer_vision.core.context import BBox, Event, FrameContext, Severity, Track
from computer_vision.inference.base import RawDetection
from computer_vision.modules.analysis.geometry import (
    Occupancy,
    ReferencePoint,
    Zone,
    ZoneError,
    ZoneStateMachine,
    parse_zones,
    reference_point,
)
from computer_vision.modules.analysis.zone import (
    EVENT_DWELL,
    EVENT_ENTRY,
    EVENT_EXIT,
    RestrictedZoneModule,
)
from computer_vision.modules.base import ModuleInitializationError, ModuleType
from computer_vision.modules.registry import ModuleDependencyError, ModuleRegistry
from tests.conftest import FakeDetectorBackend

#: Right-hand half of the frame.
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]

FRAME_W, FRAME_H = 640, 480

#: Subject keys: track ids are namespaced per tracking module.
SUBJ1 = ("person_tracking", 1)
SUBJ2 = ("person_tracking", 2)


def track(track_id: int, box, label: str = "person", conf: float = 0.9) -> Track:
    return Track(track_id=track_id, label=label, bbox=BBox(*box), confidence=conf)


def ctx(tracks, frame_id: int = 1, ts: float = 0.0, source="person_tracking") -> FrameContext:
    context = FrameContext(
        frame=np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=ts,
    )
    context.add_tracks(source, tracks)
    return context


def build(**params) -> RestrictedZoneModule:
    params.setdefault("zones", [{"name": "bay", "polygon": RIGHT_HALF}])
    module = RestrictedZoneModule(params)
    module.initialize()
    return module


#: A box entirely in the left (outside) / right (inside) half.
OUTSIDE = (10, 100, 110, 400)
INSIDE = (400, 100, 500, 400)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

class TestZone:
    def test_requires_at_least_three_points(self):
        with pytest.raises(ZoneError, match="at least 3 points"):
            Zone(name="bad", polygon=((0.0, 0.0), (1.0, 1.0)))

    def test_rejects_pixel_coordinates(self):
        """A common mistake: pasting pixel coords into a normalised field."""
        with pytest.raises(ZoneError, match="normalised"):
            Zone(name="bad", polygon=((0, 0), (640, 0), (640, 480)))

    def test_projects_onto_a_frame(self):
        zone = Zone(name="z", polygon=tuple(map(tuple, RIGHT_HALF)))
        pixels = zone.to_pixels(FRAME_W, FRAME_H)
        assert pixels.tolist() == [[320, 0], [640, 0], [640, 480], [320, 480]]

    def test_contains_respects_the_boundary(self):
        zone = Zone(name="z", polygon=tuple(map(tuple, RIGHT_HALF)))
        assert zone.contains((500, 240), FRAME_W, FRAME_H)
        assert not zone.contains((100, 240), FRAME_W, FRAME_H)
        assert zone.contains((320, 240), FRAME_W, FRAME_H)  # on the edge

    def test_survives_a_resolution_change(self):
        """The reason polygons are normalised rather than pixels."""
        zone = Zone(name="z", polygon=tuple(map(tuple, RIGHT_HALF)))
        for width, height in ((640, 480), (1280, 720), (320, 240)):
            assert zone.contains((width * 0.75, height * 0.5), width, height)
            assert not zone.contains((width * 0.25, height * 0.5), width, height)

    def test_parse_requires_a_name(self):
        with pytest.raises(ZoneError, match="needs a 'name'"):
            parse_zones([{"polygon": RIGHT_HALF}])

    def test_parse_requires_a_polygon(self):
        with pytest.raises(ZoneError, match="polygon"):
            parse_zones([{"name": "bay"}])

    def test_parse_rejects_malformed_points(self):
        with pytest.raises(ZoneError, match=r"\[x, y\] pairs"):
            parse_zones([{"name": "bay", "polygon": [[0.1, 0.1], [0.9]]}])

    def test_parse_rejects_duplicate_names(self):
        with pytest.raises(ZoneError, match="Duplicate zone names"):
            parse_zones(
                [{"name": "bay", "polygon": RIGHT_HALF}, {"name": "bay", "polygon": RIGHT_HALF}]
            )

    def test_no_zones_is_an_actionable_error(self):
        with pytest.raises(ZoneError, match="No zones configured"):
            parse_zones([])


class TestReferencePoint:
    def test_bottom_center_is_the_feet(self):
        assert reference_point(BBox(100, 50, 200, 400)) == (150, 400)

    def test_center_is_the_centroid(self):
        point = reference_point(BBox(100, 50, 200, 400), ReferencePoint.CENTER)
        assert point == (150, 225)

    def test_the_choice_changes_the_verdict(self):
        """A tall box can have its feet in and its middle out."""
        zone = Zone(name="floor", polygon=((0.0, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)))
        bbox = BBox(100, 50, 200, 400)  # centre y=225, feet y=400

        assert zone.contains(reference_point(bbox), FRAME_W, FRAME_H)
        assert not zone.contains(
            reference_point(bbox, ReferencePoint.CENTER), FRAME_W, FRAME_H
        )


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #

class TestStateMachine:
    def test_entry_is_confirmed_after_min_frames(self):
        machine = ZoneStateMachine(min_frames=2)

        assert machine.update("z", SUBJ1, True, 0.0)[0] is Occupancy.OUTSIDE
        assert machine.update("z", SUBJ1, True, 0.1)[0] is Occupancy.ENTERED

    def test_entry_fires_once(self):
        machine = ZoneStateMachine(min_frames=1)
        machine.update("z", SUBJ1, True, 0.0)

        assert machine.update("z", SUBJ1, True, 0.1)[0] is Occupancy.STILL_INSIDE
        assert machine.update("z", SUBJ1, True, 0.2)[0] is Occupancy.STILL_INSIDE

    def test_exit_is_confirmed_after_min_frames(self):
        machine = ZoneStateMachine(min_frames=2)
        machine.update("z", SUBJ1, True, 0.0)
        machine.update("z", SUBJ1, True, 0.1)

        assert machine.update("z", SUBJ1, False, 0.2)[0] is Occupancy.STILL_INSIDE
        assert machine.update("z", SUBJ1, False, 0.3)[0] is Occupancy.EXITED

    def test_boundary_jitter_does_not_flap(self):
        """Alternating in/out must not produce an entry every other frame."""
        machine = ZoneStateMachine(min_frames=3)

        transitions = [
            machine.update("z", SUBJ1, bool(i % 2), i * 0.1)[0] for i in range(12)
        ]

        assert Occupancy.ENTERED not in transitions
        assert Occupancy.EXITED not in transitions

    def test_tracks_are_independent(self):
        machine = ZoneStateMachine(min_frames=1)
        machine.update("z", SUBJ1, True, 0.0)

        assert machine.update("z", SUBJ2, True, 0.0)[0] is Occupancy.ENTERED
        assert machine.occupancy("z") == 2

    def test_zones_are_independent(self):
        machine = ZoneStateMachine(min_frames=1)
        machine.update("a", SUBJ1, True, 0.0)
        machine.update("b", SUBJ1, False, 0.0)

        assert machine.occupancy("a") == 1
        assert machine.occupancy("b") == 0

    def test_forget_reports_zones_still_occupied(self):
        machine = ZoneStateMachine(min_frames=1)
        machine.update("z", SUBJ1, True, 0.0)

        abandoned = machine.forget(SUBJ1)

        assert [name for name, _ in abandoned] == ["z"]
        assert machine.occupancy("z") == 0

    def test_forget_is_quiet_for_tracks_that_were_outside(self):
        machine = ZoneStateMachine(min_frames=1)
        machine.update("z", SUBJ1, False, 0.0)
        assert machine.forget(SUBJ1) == []


# --------------------------------------------------------------------------- #
# The module
# --------------------------------------------------------------------------- #

class TestModuleBasics:
    def test_is_an_analysis_module(self):
        assert RestrictedZoneModule.module_type is ModuleType.ANALYSIS

    def test_loads_no_model(self):
        module = build()
        assert not hasattr(module, "_backend")
        assert module.zone_names == ["bay"]

    def test_dependencies_come_from_config(self):
        assert RestrictedZoneModule.required_modules({}) == ("person_tracking",)
        assert RestrictedZoneModule.required_modules(
            {"sources": ["vehicle_tracking", "person_tracking"]}
        ) == ("vehicle_tracking", "person_tracking")

    def test_missing_zones_is_an_actionable_error(self):
        with pytest.raises(ModuleInitializationError, match="No zones configured"):
            RestrictedZoneModule({}).initialize()

    def test_bad_severity_is_rejected(self):
        with pytest.raises(ValueError, match="severity"):
            RestrictedZoneModule({"severity": "apocalyptic"})

    def test_bad_reference_point_is_rejected(self):
        with pytest.raises(ValueError, match="reference_point"):
            RestrictedZoneModule({"reference_point": "elbow"})


class TestEvents:
    def test_entry_event(self):
        module = build(min_frames=1)
        context = ctx([track(1, INSIDE)])

        module.process(context)

        events = context.get_events("restricted_zone")
        assert [e.event_type for e in events] == [EVENT_ENTRY]
        assert events[0].subject == {"track_id": 1, "class": "person"}
        assert events[0].data["zone"] == "bay"
        assert events[0].severity is Severity.WARNING

    def test_no_event_for_a_track_outside(self):
        module = build(min_frames=1)
        context = ctx([track(1, OUTSIDE)])

        module.process(context)

        assert context.events == []

    def test_entry_fires_once_not_every_frame(self):
        """The reason this consumes tracks rather than detections."""
        module = build(min_frames=1)

        totals = []
        for frame_id in range(1, 6):
            context = ctx([track(1, INSIDE)], frame_id, ts=frame_id * 0.1)
            module.process(context)
            totals.append(len(context.events))

        assert totals == [1, 0, 0, 0, 0]

    def test_exit_event_reports_dwell(self):
        module = build(min_frames=1)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))
        context = ctx([track(1, OUTSIDE)], 2, ts=4.0)

        module.process(context)

        exits = [e for e in context.events if e.event_type == EVENT_EXIT]
        assert len(exits) == 1
        assert exits[0].data["dwell_seconds"] == pytest.approx(4.0, abs=0.01)

    def test_dwell_event_fires_once_after_the_threshold(self):
        module = build(min_frames=1, dwell_seconds=2.0)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))

        before = ctx([track(1, INSIDE)], 2, ts=1.0)
        module.process(before)
        assert before.events == []

        after = ctx([track(1, INSIDE)], 3, ts=2.5)
        module.process(after)
        assert [e.event_type for e in after.events] == [EVENT_DWELL]

        later = ctx([track(1, INSIDE)], 4, ts=9.0)
        module.process(later)
        assert later.events == []

    def test_dwell_is_off_by_default(self):
        module = build(min_frames=1)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))

        context = ctx([track(1, INSIDE)], 2, ts=600.0)
        module.process(context)

        assert context.events == []

    def test_a_lost_track_closes_its_visit(self):
        """Otherwise someone the tracker loses stays 'inside' forever."""
        module = build(min_frames=1)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))

        context = ctx([], 2, ts=3.0)
        module.process(context)

        exits = [e for e in context.events if e.event_type == EVENT_EXIT]
        assert len(exits) == 1
        assert exits[0].data["reason"] == "track_lost"
        assert module.process(ctx([], 3, ts=4.0)).data["zones"]["bay"] == 0

    def test_a_missed_frame_does_not_end_the_visit(self):
        """A tracker leaves a person out on frames the detector missed them."""
        module = build(min_frames=1, dwell_seconds=5)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))

        gap = ctx([], 2, ts=0.5)
        result = module.process(gap)
        assert gap.events == []
        assert result.data["zones"]["bay"] == 1

        back = ctx([track(1, INSIDE)], 3, ts=1.0)
        module.process(back)
        assert back.events == []

        later = ctx([track(1, INSIDE)], 4, ts=6.0)
        module.process(later)
        assert [e.event_type for e in later.events] == [EVENT_DWELL]

    def test_a_track_gone_longer_than_the_grace_exits_from_when_last_seen(self):
        module = build(min_frames=1, lost_grace_seconds=2.0)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))
        module.process(ctx([track(1, INSIDE)], 2, ts=4.0))
        assert module.process(ctx([], 3, ts=5.0)).data["zones"]["bay"] == 1

        context = ctx([], 4, ts=6.5)
        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_EXIT]
        assert context.events[0].data["dwell_seconds"] == 4.0
        assert context.events[0].data["reason"] == "track_lost"

    def test_without_a_grace_a_lost_track_exits_at_once(self):
        module = build(min_frames=1, lost_grace_seconds=0)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))

        context = ctx([], 2, ts=0.5)
        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_EXIT]
        assert context.events[0].data["dwell_seconds"] == 0.5

    def test_a_new_id_in_the_same_place_continues_the_visit(self):
        """The tracker renumbered someone: one visit, not an exit and an entry."""
        module = build(min_frames=1, dwell_seconds=5)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))
        module.process(ctx([], 2, ts=0.5))

        nearby = (410, 105, 510, 405)
        renamed = ctx([track(2, nearby)], 3, ts=1.0)
        result = module.process(renamed)
        assert renamed.events == []
        assert result.data["zones"]["bay"] == 1

        later = ctx([track(2, nearby)], 4, ts=5.5)
        module.process(later)
        assert [e.event_type for e in later.events] == [EVENT_DWELL]
        assert later.events[0].subject["track_id"] == 2

        gone = ctx([], 5, ts=9.0)
        module.process(gone)
        assert [e.event_type for e in gone.events] == [EVENT_EXIT]
        assert gone.events[0].data["dwell_seconds"] == 5.5

    def test_a_new_id_far_away_is_a_new_visit(self):
        module = build(min_frames=1, zones=[{"name": "bay", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
        module.process(ctx([track(1, OUTSIDE)], 1, ts=0.0))

        context = ctx([track(2, INSIDE)], 2, ts=0.5)
        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_ENTRY]
        assert context.events[0].subject["track_id"] == 2

    def test_a_new_id_does_not_take_over_a_track_that_is_still_there(self):
        module = build(min_frames=1)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))

        context = ctx([track(1, INSIDE), track(2, (410, 105, 510, 405))], 2, ts=0.5)
        result = module.process(context)

        assert [e.subject["track_id"] for e in context.events] == [2]
        assert result.data["zones"]["bay"] == 2

    def test_a_different_kind_of_thing_does_not_continue_the_visit(self):
        module = build(min_frames=1)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))
        module.process(ctx([], 2, ts=0.5))

        context = ctx([track(2, INSIDE, label="dog")], 3, ts=1.0)
        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_ENTRY]

    def test_handoff_can_be_turned_off(self):
        module = build(min_frames=1, handoff_distance=0)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))
        module.process(ctx([], 2, ts=0.5))

        context = ctx([track(2, INSIDE)], 3, ts=1.0)
        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_ENTRY]

    def test_occupancy_is_reported(self):
        module = build(min_frames=1)
        context = ctx([track(1, INSIDE), track(2, INSIDE), track(3, OUTSIDE)])

        result = module.process(context)

        assert result.data["zones"] == {"bay": 2}
        assert result.data["occupied"] == ["bay"]

    def test_event_totals_accumulate(self):
        module = build(min_frames=1)
        module.process(ctx([track(1, INSIDE)], 1, ts=0.0))
        module.process(ctx([track(1, OUTSIDE)], 2, ts=1.0))

        assert module.event_totals == {EVENT_ENTRY: 1, EVENT_EXIT: 1}

    def test_severity_is_configurable(self):
        module = build(min_frames=1, severity="critical")
        context = ctx([track(1, INSIDE)])

        module.process(context)

        assert context.events[0].severity is Severity.CRITICAL

    def test_class_allowlist_filters_tracks(self):
        module = build(min_frames=1, classes=["person"])
        context = ctx([track(1, INSIDE, label="dog")])

        module.process(context)

        assert context.events == []

    def test_watches_several_sources(self):
        module = build(min_frames=1, sources=["person_tracking", "vehicle_tracking"])
        context = ctx([track(1, INSIDE)], source="person_tracking")
        context.add_tracks("vehicle_tracking", [track(1, INSIDE, label="car")])

        module.process(context)

        assert len(context.events) == 2
        assert {e.subject["class"] for e in context.events} == {"person", "car"}

    def test_ignores_untracked_sources(self):
        module = build(min_frames=1, sources=["person_tracking"])
        context = ctx([], source="person_tracking")
        context.add_tracks("vehicle_tracking", [track(1, INSIDE, label="car")])

        module.process(context)

        assert context.events == []

    def test_multiple_zones_report_separately(self):
        module = build(
            min_frames=1,
            zones=[
                {"name": "left", "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]},
                {"name": "right", "polygon": RIGHT_HALF},
            ],
        )
        context = ctx([track(1, OUTSIDE), track(2, INSIDE)])

        result = module.process(context)

        assert result.data["zones"] == {"left": 1, "right": 1}
        assert {e.data["zone"] for e in context.events} == {"left", "right"}


class TestOverlays:
    def test_zones_are_published_for_drawing(self):
        module = build(min_frames=1)
        context = ctx([])

        module.process(context)

        assert len(context.overlays) == 1
        overlay = context.overlays[0]
        assert overlay.kind == "polygon"
        assert overlay.label == "bay"
        assert overlay.active is False
        assert len(overlay.points) == 4

    def test_an_occupied_zone_is_marked_active(self):
        module = build(min_frames=1)
        context = ctx([track(1, INSIDE)])

        module.process(context)

        overlay = context.overlays[0]
        assert overlay.active is True
        assert "(1)" in overlay.label

    def test_overlay_points_follow_the_frame_size(self):
        module = build(min_frames=1)
        context = FrameContext(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            frame_id=1,
            source_id="t",
            monotonic_ts=0.0,
        )
        module.process(context)

        assert context.overlays[0].points[0] == (640, 0)


# --------------------------------------------------------------------------- #
# Event plumbing in the context
# --------------------------------------------------------------------------- #

class TestContextEvents:
    def test_events_are_namespaced_by_producer(self):
        context = ctx([])
        context.add_event(
            Event(event_type="a", source_module="restricted_zone", frame_id=1, timestamp="t")
        )
        context.add_event(
            Event(event_type="b", source_module="other", frame_id=1, timestamp="t")
        )

        assert len(context.get_events("restricted_zone")) == 1
        assert len(context.get_events()) == 2

    def test_events_reach_the_summary(self):
        import json

        module = build(min_frames=1)
        context = ctx([track(1, INSIDE)])
        module.process(context)

        summary = context.summary()
        json.dumps(summary)

        assert summary["events"][0]["event"] == EVENT_ENTRY
        assert summary["events"][0]["data"]["zone"] == "bay"

    def test_describe_is_human_readable(self):
        event = Event(
            event_type=EVENT_ENTRY,
            source_module="restricted_zone",
            frame_id=1,
            timestamp="t",
            subject={"track_id": 3, "class": "person"},
            data={"zone": "bay"},
        )
        assert event.describe() == "zone_entry: person #3 in 'bay'"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

class TestRegistryWiring:
    def test_resolves_by_name(self):
        assert ModuleRegistry.resolve("restricted_zone") is RestrictedZoneModule

    def test_runs_after_its_tracker(self):
        built = ModuleRegistry.build_enabled(
            build_config(
                {
                    "modules": {
                        "restricted_zone": {
                            "enabled": True,
                            "zones": [{"name": "bay", "polygon": RIGHT_HALF}],
                        },
                        "person_detection": {"enabled": True, "backend": "fake"},
                        "person_tracking": {"enabled": True},
                    }
                }
            )
        )
        order = [m.module_name for m in built]
        assert order.index("person_tracking") < order.index("restricted_zone")
        assert order.index("person_detection") < order.index("person_tracking")

    def test_configured_source_becomes_the_dependency(self):
        """Watching vehicles must require vehicle_tracking, not person_tracking."""
        with pytest.raises(ModuleDependencyError, match="vehicle_tracking"):
            ModuleRegistry.build_enabled(
                build_config(
                    {
                        "modules": {
                            "restricted_zone": {
                                "enabled": True,
                                "sources": ["vehicle_tracking"],
                                "zones": [{"name": "bay", "polygon": RIGHT_HALF}],
                            },
                            "person_detection": {"enabled": True, "backend": "fake"},
                            "person_tracking": {"enabled": True},
                        }
                    }
                )
            )

    def test_without_any_tracker_it_is_rejected(self):
        with pytest.raises(ModuleDependencyError, match="person_tracking"):
            ModuleRegistry.build_enabled(
                build_config(
                    {
                        "modules": {
                            "restricted_zone": {
                                "enabled": True,
                                "zones": [{"name": "bay", "polygon": RIGHT_HALF}],
                            }
                        }
                    }
                )
            )


class TestFullPipeline:
    def test_detection_to_tracking_to_event(self):
        """The whole chain: pixels in, a zone event out."""
        from computer_vision.core.frame_processor import FrameProcessor
        from computer_vision.core.pipeline import VisionPipeline
        from tests.test_pipeline import RecordingSink, SyntheticSource

        # A person standing in the right-hand half of a 640x480 frame.
        FakeDetectorBackend.scripted = [
            RawDetection(label="person", confidence=0.9, xyxy=(400, 100, 500, 400), class_id=0)
        ]
        config = build_config(
            {
                "processing": {"show_window": False, "target_width": None},
                "modules": {
                    "person_detection": {"enabled": True, "backend": "fake"},
                    "person_tracking": {"enabled": True, "min_hits": 1},
                    "restricted_zone": {
                        "enabled": True,
                        "min_frames": 1,
                        "zones": [{"name": "bay", "polygon": RIGHT_HALF}],
                    },
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

        all_events = [e for c in sink.contexts for e in c.events]
        assert [e.event_type for e in all_events] == [EVENT_ENTRY]
        assert all_events[0].subject["track_id"] == 1
        assert sink.contexts[-1].get_result("restricted_zone").data["zones"] == {"bay": 1}

    @pytest.mark.parametrize("size", [(640, 480), (1280, 720), (320, 240)])
    def test_zone_events_survive_a_resolution_change(self, size):
        """Normalised polygons mean target_width does not move the zone.

        The track is placed at the same *fraction* of the frame each time,
        which is what a detector running on a resized frame actually produces.
        """
        width, height = size
        module = build(min_frames=1)

        context = FrameContext(
            frame=np.zeros((height, width, 3), dtype=np.uint8),
            frame_id=1,
            source_id="t",
            monotonic_ts=0.0,
        )
        # Standing at 75% across, feet at 85% down - inside the right half.
        context.add_tracks(
            "person_tracking",
            [
                track(
                    1,
                    (width * 0.70, height * 0.2, width * 0.80, height * 0.85),
                )
            ],
        )

        module.process(context)

        assert [e.event_type for e in context.events] == [EVENT_ENTRY]

    def test_a_track_outside_never_fires_at_any_resolution(self):
        for width, height in ((640, 480), (1280, 720)):
            module = build(min_frames=1)
            context = FrameContext(
                frame=np.zeros((height, width, 3), dtype=np.uint8),
                frame_id=1,
                source_id="t",
                monotonic_ts=0.0,
            )
            context.add_tracks(
                "person_tracking",
                [track(1, (width * 0.05, height * 0.2, width * 0.20, height * 0.85))],
            )

            module.process(context)

            assert context.events == []
