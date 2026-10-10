"""FrameContext, BBox, Detection and Track behaviour."""

from __future__ import annotations

import pytest

from computer_vision.core.context import BBox, Detection, FrameContext, Track


class TestBBox:
    def test_geometry(self):
        box = BBox(10, 20, 110, 220)
        assert box.width == 100
        assert box.height == 200
        assert box.area == 20000
        assert box.center == (60, 120)
        assert box.as_int() == (10, 20, 110, 220)

    def test_rejects_inverted_box(self):
        with pytest.raises(ValueError, match="Invalid bbox"):
            BBox(100, 10, 10, 100)

    def test_clip_to_frame(self):
        clipped = BBox(-20, -10, 900, 700).clip(640, 480)
        assert clipped.as_tuple() == (0.0, 0.0, 640.0, 480.0)

    def test_iou(self):
        a = BBox(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(BBox(20, 20, 30, 30)) == 0.0
        assert a.iou(BBox(5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_constructors(self):
        assert BBox.from_xyxy([1, 2, 3, 4]).as_tuple() == (1.0, 2.0, 3.0, 4.0)
        assert BBox.from_xywh(1, 2, 10, 20).as_tuple() == (1.0, 2.0, 11.0, 22.0)


class TestSchemas:
    def test_detection_schema_matches_spec(self):
        det = Detection(label="person", confidence=0.9412, bbox=BBox(1, 2, 3, 4))
        assert det.to_dict() == {
            "class": "person",
            "confidence": 0.9412,
            "bbox": [1.0, 2.0, 3.0, 4.0],
        }

    def test_track_schema_matches_spec(self):
        track = Track(track_id=1, label="person", bbox=BBox(1, 2, 3, 4), confidence=0.91)
        payload = track.to_dict()
        assert payload["track_id"] == 1
        assert payload["class"] == "person"
        assert payload["confidence"] == 0.91
        assert payload["bbox"] == [1.0, 2.0, 3.0, 4.0]


class TestFrameContext:
    def test_dimensions_come_from_the_frame(self, frame_context: FrameContext):
        assert frame_context.shape == (640, 480)

    def test_detections_are_namespaced_by_producer(self, frame_context: FrameContext):
        frame_context.add_detections(
            "person_detection", [Detection("person", 0.9, BBox(0, 0, 10, 10))]
        )
        frame_context.add_detections(
            "ppe_detection", [Detection("helmet", 0.8, BBox(0, 0, 5, 5))]
        )

        assert len(frame_context.get_detections("person_detection")) == 1
        assert len(frame_context.get_detections("ppe_detection")) == 1
        assert len(frame_context.get_detections()) == 2
        assert frame_context.get_detections("fire_detection") == []

    def test_producer_is_stamped_on_detections(self, frame_context: FrameContext):
        frame_context.add_detections(
            "person_detection", [Detection("person", 0.9, BBox(0, 0, 10, 10))]
        )
        assert frame_context.get_detections()[0].source_module == "person_detection"

    def test_detection_then_tracking_handoff(self, frame_context: FrameContext):
        """The exact data path Phase 2 -> Phase 3 relies on."""
        frame_context.add_detections(
            "person_detection", [Detection("person", 0.9, BBox(0, 0, 10, 10))]
        )

        # A tracking module reads its dependency's output by name...
        detections = frame_context.get_detections("person_detection")
        assert len(detections) == 1

        # ...and publishes tracks under its own name.
        frame_context.add_tracks(
            "person_tracking",
            [Track(track_id=1, label="person", bbox=detections[0].bbox, confidence=0.9)],
        )
        assert frame_context.get_tracks("person_tracking")[0].track_id == 1

    def test_get_detections_returns_a_copy(self, frame_context: FrameContext):
        frame_context.add_detections(
            "person_detection", [Detection("person", 0.9, BBox(0, 0, 10, 10))]
        )
        frame_context.get_detections("person_detection").clear()
        assert len(frame_context.get_detections("person_detection")) == 1

    def test_errors_are_recorded(self, frame_context: FrameContext):
        frame_context.record_error("fire_detection", "model missing")
        assert frame_context.errors["fire_detection"] == "model missing"

    def test_summary_is_json_safe(self, frame_context: FrameContext):
        import json

        frame_context.add_detections(
            "person_detection", [Detection("person", 0.9, BBox(0, 0, 10, 10))]
        )
        frame_context.metadata["fps"] = 30.0

        summary = frame_context.summary()
        json.dumps(summary)  # must not raise

        assert "frame" not in summary  # never ship pixels in a result payload
        assert summary["frame_id"] == 1
        assert summary["detections"]["person_detection"][0]["class"] == "person"
