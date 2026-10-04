"""Live-view decoupling, the direct OpenVINO backend, and the tracking and
false-detection fixes that came with them.

Like the rest of the suite: no camera, no display, no model. The one test that
needs the real model and OpenVINO is marked ``integration``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from backend.api import server as server_module
from backend.services.state import DashboardState
from backend.services.web_sink import WebSink
from computer_vision.core.config import DashboardConfig, ProcessingConfig
from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.core.frame_processor import FrameProcessor
from computer_vision.inference.base import ModelLoadError, RawDetection
from computer_vision.inference.factory import available_backends
from computer_vision.inference.openvino_backend import (
    OpenVinoDetector,
    letterbox,
    resolve_device,
)
from computer_vision.modules.analysis.zone import RestrictedZoneModule
from computer_vision.modules.detection.object import ObjectDetectionModule
from computer_vision.modules.tracking.person import PersonTrackingModule
from computer_vision.modules.tracking.tracker_core import IoUTracker
from computer_vision.sources.rtsp import RTSPSource
from computer_vision.utils.timing import FpsMeter, LatencyMeter, PipelineStats
from tests.conftest import FakeDetectorBackend
from tests.test_rtsp_source import fake_stream, rtsp_config  # noqa: F401 - fixture

FRAME_W, FRAME_H = 640, 480
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


def det(x1, y1, x2, y2, label="person", conf=0.9) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(x1, y1, x2, y2))


def context(frame_id=1, ts=0.0) -> FrameContext:
    return FrameContext(
        frame=np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=ts,
    )


def stats() -> PipelineStats:
    return PipelineStats(
        fps_meter=FpsMeter(5), total_latency=LatencyMeter(5), capture_latency=LatencyMeter(5)
    )


def wait_for(condition, timeout=3.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# Tracker: motion prediction, distance matching, ids, travel
# --------------------------------------------------------------------------- #


class TestMotionPrediction:
    def test_a_moving_track_is_predicted_ahead_of_its_last_box(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1)
        tracker.update([det(0, 100, 40, 200)])
        tracker.update([det(15, 100, 55, 200)])  # 15 px to the right

        predicted = tracker.active_tracks[0].predicted_bbox()
        assert predicted.x1 > 15  # moved on from the last box, not sitting on it

    def test_a_walker_keeps_their_id_when_steps_approach_the_box_width(self):
        """40 px wide, 36 px per frame: the previous box overlaps by ~0.05,
        well under the threshold. Only the predicted box still matches."""
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1)
        ids = set()
        for x in (0, 25, 50, 86, 122, 158, 194, 230):  # speeds up after two frames
            ids.update(t.track_id for t in tracker.update([det(x, 100, x + 40, 200)]))
        assert ids == {1}

    def test_a_track_seen_once_predicts_its_own_box(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(10, 10, 60, 200)])
        track = tracker.active_tracks[0]
        assert track.predicted_bbox() == track.bbox


class TestDistanceMatching:
    def test_off_by_default(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1)
        tracker.update([det(0, 100, 40, 200)])
        tracks = tracker.update([det(60, 100, 100, 200)])  # no overlap
        assert [t.track_id for t in tracks] == [2]

    def test_a_near_detection_with_no_overlap_continues_the_track(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1, max_center_distance=0.8)
        tracker.update([det(0, 100, 40, 200)])
        tracks = tracker.update([det(60, 100, 100, 200)])  # 60 px = 0.6 of its height
        assert [t.track_id for t in tracks] == [1]

    def test_a_far_detection_starts_a_new_track(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1, max_center_distance=0.8)
        tracker.update([det(0, 100, 40, 200)])
        tracks = tracker.update([det(300, 100, 340, 200)])
        assert [t.track_id for t in tracks] == [2]

    def test_a_box_of_a_very_different_size_is_not_the_same_thing(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1, max_center_distance=0.8)
        tracker.update([det(0, 100, 40, 200)])
        tracks = tracker.update([det(60, 140, 70, 165)])  # near, but tiny
        assert [t.track_id for t in tracks] == [2]

    def test_labels_are_still_respected(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1, max_center_distance=0.8)
        tracker.update([det(0, 100, 40, 200, label="person")])
        tracks = tracker.update([det(60, 100, 100, 200, label="dog")])
        assert [t.label for t in tracks] == ["dog"]
        assert tracks[0].track_id == 2

    def test_an_overlapping_match_is_never_taken_away(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1, max_center_distance=2.0)
        tracker.update([det(0, 100, 40, 200), det(200, 100, 240, 200)])
        tracks = tracker.update([det(5, 100, 45, 200), det(205, 100, 245, 200)])
        assert [t.track_id for t in tracks] == [1, 2]
        assert tracks[0].bbox.x1 == 5 and tracks[1].bbox.x1 == 205

    def test_a_track_unseen_for_a_while_is_not_handed_to_a_neighbour(self):
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1, max_center_distance=0.8)
        tracker.update([det(0, 100, 40, 200)])
        for _ in range(3):
            tracker.update([])
        tracks = tracker.update([det(60, 100, 100, 200)])
        assert [t.track_id for t in tracks] == [2]

    def test_rejects_a_negative_distance(self):
        with pytest.raises(ValueError):
            IoUTracker(max_center_distance=-1)


class TestCompactIds:
    def test_default_numbers_every_first_sighting(self):
        tracker = IoUTracker(min_hits=2)
        tracker.update([det(0, 0, 40, 100)])
        tracker.update([])
        assert tracker.total_ids_issued == 1

    def test_unconfirmed_flickers_use_no_ids(self):
        tracker = IoUTracker(min_hits=2, max_age=0, compact_ids=True)
        for step in range(5):  # five one-frame detections, each somewhere else
            tracker.update([det(step * 100, 0, step * 100 + 40, 100)])
        assert tracker.total_ids_issued == 0

        tracker.update([det(0, 300, 40, 400)])
        tracks = tracker.update([det(2, 300, 42, 400)])
        assert [t.track_id for t in tracks] == [1]

    def test_ids_are_consecutive_in_order_of_confirmation(self):
        tracker = IoUTracker(min_hits=2, compact_ids=True)
        boxes = [det(0, 0, 40, 100), det(300, 0, 340, 100)]
        tracker.update(boxes)
        tracks = tracker.update(boxes)
        assert sorted(t.track_id for t in tracks) == [1, 2]

    def test_nothing_reported_ever_has_id_zero(self):
        tracker = IoUTracker(min_hits=3, compact_ids=True)
        for _ in range(6):
            for track in tracker.update([det(0, 0, 40, 100)]):
                assert track.track_id > 0

    def test_min_hits_one_numbers_immediately(self):
        tracker = IoUTracker(min_hits=1, compact_ids=True)
        assert [t.track_id for t in tracker.update([det(0, 0, 40, 100)])] == [1]


class TestTravel:
    def test_something_that_never_moves_has_no_travel(self):
        tracker = IoUTracker(min_hits=1)
        for jitter in (0, 2, -1, 1, 0):
            tracks = tracker.update([det(100 + jitter, 100, 140 + jitter, 200)])
        assert tracks[0].travel < 0.1
        assert tracks[0].as_track().extra["travel"] < 0.1

    def test_a_walker_accumulates_travel(self):
        tracker = IoUTracker(min_hits=1)
        for step in range(10):
            tracks = tracker.update([det(step * 20, 100, step * 20 + 40, 200)])
        assert tracks[0].travel > 1.0

    def test_travel_does_not_shrink_when_the_walker_returns(self):
        tracker = IoUTracker(min_hits=1)
        for x in (0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0):
            tracks = tracker.update([det(x, 100, x + 40, 200)])
        assert tracks[0].travel == pytest.approx(0.5)  # 50 px of a 100 px box, kept


class TestTrackingModuleAdaptsToFrameRate:
    def _run(self, gap_seconds: float) -> PersonTrackingModule:
        module = PersonTrackingModule({"max_center_distance": 0.8, "min_hits": 1})
        module.initialize()
        for index in range(4):
            ctx = context(index + 1, ts=index * gap_seconds)
            ctx.add_detections("person_detection", [det(0, 100, 40, 200)])
            module.process(ctx)
        return module

    def test_distance_matching_is_off_at_a_normal_frame_rate(self):
        assert self._run(1 / 15)._tracker.max_center_distance == 0.0

    def test_distance_matching_is_on_when_frames_are_far_apart(self):
        assert self._run(0.6)._tracker.max_center_distance == 0.8

    def test_without_the_option_nothing_changes(self):
        module = PersonTrackingModule({"min_hits": 1})
        module.initialize()
        for index in range(4):
            ctx = context(index + 1, ts=index * 0.6)
            ctx.add_detections("person_detection", [det(0, 100, 40, 200)])
            module.process(ctx)
        assert module._tracker.max_center_distance == 0.0

    def test_tracks_report_their_travel(self):
        module = self._run(0.6)
        ctx = context(9, ts=3.0)
        ctx.add_detections("person_detection", [det(0, 100, 40, 200)])
        module.process(ctx)
        assert "travel" in ctx.get_tracks("person_tracking")[0].extra


# --------------------------------------------------------------------------- #
# Restricted zone: ignoring what never moved
# --------------------------------------------------------------------------- #


class TestZoneMinTravel:
    INSIDE = (400, 100, 460, 300)

    def _events(self, module, tracks, frames=4):
        events = []
        for index in range(frames):
            ctx = context(index + 1, ts=float(index))
            ctx.add_tracks("person_tracking", tracks)
            module.process(ctx)
            events.extend(e.event_type for e in ctx.events)
        return events

    def _module(self, **params) -> RestrictedZoneModule:
        module = RestrictedZoneModule(
            {"zones": [{"name": "bay", "polygon": RIGHT_HALF}], "dwell_seconds": 2, **params}
        )
        module.initialize()
        return module

    def _track(self, travel=None) -> Track:
        extra = {} if travel is None else {"travel": travel}
        return Track(
            track_id=1, label="person", bbox=BBox(*self.INSIDE), confidence=0.9, extra=extra
        )

    def test_default_reports_a_static_track(self):
        assert "zone_entry" in self._events(self._module(), [self._track(travel=0.0)])

    def test_a_track_that_never_moved_raises_nothing(self):
        module = self._module(min_travel=0.5)
        assert self._events(module, [self._track(travel=0.1)]) == []
        assert module.process(context(9, ts=9.0)).data["zones"]["bay"] == 0

    def test_a_track_that_walked_in_is_reported(self):
        events = self._events(self._module(min_travel=0.5), [self._track(travel=3.0)])
        assert events.count("zone_entry") == 1
        assert "zone_dwell" in events

    def test_a_static_track_is_reported_once_it_moves(self):
        module = self._module(min_travel=0.5)
        assert self._events(module, [self._track(travel=0.1)]) == []
        assert "zone_entry" in self._events(module, [self._track(travel=0.9)])

    def test_a_track_without_travel_information_is_taken_at_face_value(self):
        assert "zone_entry" in self._events(self._module(min_travel=0.5), [self._track()])


# --------------------------------------------------------------------------- #
# Detection: duplicates
# --------------------------------------------------------------------------- #


def raw(label, conf, box) -> RawDetection:
    return RawDetection(label=label, confidence=conf, xyxy=box)


class TestDuplicateSuppression:
    def _labels(self, scripted, **params):
        FakeDetectorBackend.scripted = scripted
        FakeDetectorBackend.scripted_sequence = None
        module = ObjectDetectionModule({"backend": "fake", **params})
        module.initialize()
        ctx = context()
        module.process(ctx)
        return [(d.label, round(d.confidence, 2)) for d in ctx.get_detections("object_detection")]

    def test_everything_is_reported_by_default(self):
        scripted = [raw("person", 0.9, (0, 0, 50, 100)), raw("chair", 0.8, (200, 0, 250, 100))]
        assert self._labels(scripted) == [("person", 0.9), ("chair", 0.8)]

    def test_excluded_labels_are_not_reported(self):
        scripted = [raw("person", 0.9, (0, 0, 50, 100)), raw("chair", 0.8, (200, 0, 250, 100))]
        assert self._labels(scripted, exclude_labels=["person", "car"]) == [("chair", 0.8)]

    def test_two_classes_on_one_spot_become_the_more_confident_one(self):
        scripted = [
            raw("suitcase", 0.6, (100, 100, 200, 200)),
            raw("chair", 0.8, (102, 101, 201, 199)),
        ]
        assert self._labels(scripted, dedupe_iou=0.7) == [("chair", 0.8)]

    def test_two_of_the_same_class_are_left_to_the_models_own_nms(self):
        scripted = [
            raw("chair", 0.6, (100, 100, 200, 200)),
            raw("chair", 0.8, (102, 101, 201, 199)),
        ]
        assert len(self._labels(scripted, dedupe_iou=0.7)) == 2

    def test_different_classes_that_only_touch_are_both_kept(self):
        scripted = [
            raw("backpack", 0.6, (100, 100, 140, 150)),
            raw("chair", 0.8, (100, 100, 220, 260)),
        ]
        assert len(self._labels(scripted, dedupe_iou=0.7)) == 2

    def test_dedupe_is_off_by_default(self):
        scripted = [
            raw("suitcase", 0.6, (100, 100, 200, 200)),
            raw("chair", 0.8, (102, 101, 201, 199)),
        ]
        assert len(self._labels(scripted)) == 2


# --------------------------------------------------------------------------- #
# OpenVINO backend (the parts that need no runtime)
# --------------------------------------------------------------------------- #


class TestLetterbox:
    def test_a_16_9_frame_gets_two_rows_of_padding_at_544x960(self):
        frame = np.full((540, 960, 3), 200, dtype=np.uint8)
        image, scale, (left, top) = letterbox(frame, 544, 960)
        assert image.shape == (544, 960, 3)
        assert scale == 1.0 and (left, top) == (0, 2)
        assert image[0, 0, 0] == 114 and image[2, 0, 0] == 200

    def test_a_larger_frame_is_scaled_down_keeping_its_shape(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image, scale, (left, top) = letterbox(frame, 384, 640)
        assert image.shape == (384, 640, 3)
        assert scale == pytest.approx(640 / 1920)
        assert (left, top) == (0, 12)

    def test_an_exact_fit_is_returned_untouched(self):
        frame = np.zeros((544, 960, 3), dtype=np.uint8)
        image, scale, offset = letterbox(frame, 544, 960)
        assert image is frame and scale == 1.0 and offset == (0, 0)


class TestOpenVinoDecode:
    def _detector(self, **params) -> OpenVinoDetector:
        detector = OpenVinoDetector({"model": "unused", **params})
        detector._names = {0: "person", 1: "bicycle", 2: "car"}
        return detector

    def _output(self, boxes):
        """Model output for ``(cx, cy, w, h, class_id, score)`` rows, padded with noise."""
        output = np.zeros((4 + 3, len(boxes) + 5), dtype=np.float32)
        for column, (cx, cy, w, h, class_id, score) in enumerate(boxes):
            output[:4, column] = (cx, cy, w, h)
            output[4 + class_id, column] = score
        output[4:, len(boxes):] = 0.01  # below any threshold
        return output

    def test_boxes_are_mapped_back_through_the_letterbox(self):
        # 1920x1080 frame in a 384x640 input: scale 1/3, 12 rows padding on top.
        detections = self._detector()._decode(
            self._output([(320, 192, 60, 120, 0, 0.9)]), 1 / 3, 0, 12, 1920, 1080
        )
        assert len(detections) == 1
        only = detections[0]
        assert only.label == "person" and only.class_id == 0
        assert only.confidence == pytest.approx(0.9)
        assert only.xyxy == pytest.approx((870, 360, 1050, 720), abs=0.01)

    def test_overlapping_boxes_of_one_class_are_suppressed(self):
        detections = self._detector()._decode(
            self._output([(100, 100, 50, 80, 0, 0.9), (102, 101, 50, 80, 0, 0.6)]),
            1.0, 0, 0, 960, 540,
        )
        assert [round(d.confidence, 2) for d in detections] == [0.9]

    def test_overlapping_boxes_of_different_classes_are_both_kept(self):
        detections = self._detector()._decode(
            self._output([(100, 100, 50, 80, 0, 0.9), (102, 101, 50, 80, 2, 0.6)]),
            1.0, 0, 0, 960, 540,
        )
        assert [d.label for d in detections] == ["person", "car"]

    def test_scores_below_the_model_confidence_are_dropped(self):
        detections = self._detector(model_confidence=0.5)._decode(
            self._output([(100, 100, 50, 80, 0, 0.4), (400, 100, 50, 80, 2, 0.7)]),
            1.0, 0, 0, 960, 540,
        )
        assert [d.label for d in detections] == ["car"]

    def test_class_restriction(self):
        detections = self._detector(classes=[2])._decode(
            self._output([(100, 100, 50, 80, 0, 0.9), (400, 100, 50, 80, 2, 0.7)]),
            1.0, 0, 0, 960, 540,
        )
        assert [d.label for d in detections] == ["car"]

    def test_boxes_are_clipped_to_the_frame(self):
        detections = self._detector()._decode(
            self._output([(10, 10, 100, 100, 0, 0.9)]), 1.0, 0, 0, 960, 540
        )
        assert detections[0].xyxy[:2] == (0.0, 0.0)

    def test_nothing_found_is_an_empty_list(self):
        assert self._detector()._decode(self._output([]), 1.0, 0, 0, 960, 540) == []


class TestOpenVinoBackendSetup:
    def test_is_a_registered_backend(self):
        assert "openvino" in available_backends()

    @pytest.mark.parametrize(
        "given, expected",
        [("intel:cpu", "CPU"), ("cpu", "CPU"), ("intel:gpu", "GPU"), ("GPU", "GPU"),
         ("auto", "AUTO"), ("GPU.1", "GPU.1")],
    )
    def test_device_names(self, given, expected):
        assert resolve_device(given) == expected

    def test_infer_before_load_is_an_error(self):
        with pytest.raises(ModelLoadError):
            OpenVinoDetector({"model": "x"}).infer(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_a_pt_file_is_refused_with_a_pointer_to_the_other_backend(self, tmp_path):
        pytest.importorskip("openvino")
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"not a model")
        with pytest.raises(ModelLoadError, match="backend: ultralytics"):
            OpenVinoDetector({"model": str(weights)}).load()

    def test_a_missing_model_is_an_error_not_a_download(self, tmp_path):
        pytest.importorskip("openvino")
        with pytest.raises(ModelLoadError):
            OpenVinoDetector({"model": str(tmp_path / "nope_openvino_model" / "m.xml")}).load()


class _FakeTensor:
    def __init__(self, data):
        self.data = data


class _FakeRequest:
    """Stands in for an OpenVINO inference request on one device."""

    def __init__(self, seconds: float, scores: np.ndarray, fail: bool = False):
        self.seconds, self.scores, self.fail = seconds, scores, fail

    def infer(self, _inputs):
        if self.fail:
            raise RuntimeError("driver error")
        time.sleep(self.seconds)

    def get_output_tensor(self, _index):
        boxes = np.zeros((4, self.scores.shape[1]), dtype=np.float32)
        return _FakeTensor(np.concatenate([boxes, self.scores])[None])


class TestAutoDevice:
    """`device: intel:auto` - the graphics chip only when it earns it."""

    SCORES = np.full((3, 8), 0.1, dtype=np.float32)

    def _choose(self, monkeypatch, available, gpu=None, cpu_seconds=0.02, compile_fails=False):
        detector = OpenVinoDetector({"model": "x", "device": "intel:auto"})
        detector._height, detector._width = 64, 96
        compiled = []

        def compile_(core, model, device, cache_dir):
            compiled.append(device)
            if device == "GPU":
                if compile_fails:
                    raise RuntimeError("no OpenCL")
                return gpu
            return _FakeRequest(cpu_seconds, self.SCORES)

        monkeypatch.setattr(detector, "_compile", compile_)
        request, device = detector._fastest(None, None, available, "")
        return request, device, compiled

    def test_without_a_graphics_device_it_is_the_cpu(self, monkeypatch):
        _, device, compiled = self._choose(monkeypatch, ["CPU"])
        assert device == "CPU" and compiled == ["CPU"]

    def test_a_clearly_faster_graphics_chip_is_used(self, monkeypatch):
        gpu = _FakeRequest(0.002, self.SCORES)
        request, device, _ = self._choose(monkeypatch, ["CPU", "GPU.0"], gpu)
        assert device == "GPU" and request is gpu

    def test_a_graphics_chip_that_is_not_faster_is_not_used(self, monkeypatch):
        gpu = _FakeRequest(0.02, self.SCORES)
        _, device, _ = self._choose(monkeypatch, ["CPU", "GPU"], gpu, cpu_seconds=0.02)
        assert device == "CPU"

    def test_a_graphics_chip_with_different_results_is_not_used(self, monkeypatch):
        gpu = _FakeRequest(0.002, self.SCORES + 0.3)
        _, device, _ = self._choose(monkeypatch, ["CPU", "GPU"], gpu)
        assert device == "CPU"

    def test_a_graphics_chip_returning_nonsense_is_not_used(self, monkeypatch):
        gpu = _FakeRequest(0.002, np.full((3, 8), np.nan, dtype=np.float32))
        _, device, _ = self._choose(monkeypatch, ["CPU", "GPU"], gpu)
        assert device == "CPU"

    def test_a_driver_that_cannot_compile_falls_back_to_the_cpu(self, monkeypatch):
        _, device, _ = self._choose(monkeypatch, ["CPU", "GPU"], compile_fails=True)
        assert device == "CPU"

    def test_a_driver_that_fails_when_run_falls_back_to_the_cpu(self, monkeypatch):
        gpu = _FakeRequest(0.002, self.SCORES, fail=True)
        _, device, _ = self._choose(monkeypatch, ["CPU", "GPU"], gpu)
        assert device == "CPU"


MODEL_DIR = Path("computer_vision/Model/detection/yolo11s_int8_openvino_model")


@pytest.mark.integration
class TestAutoDeviceWithTheRealModel:
    def test_auto_loads_and_detects_like_the_cpu(self):
        pytest.importorskip("openvino")
        if not MODEL_DIR.is_dir():
            pytest.skip("default model not present")
        frame = np.random.default_rng(1).integers(0, 255, (540, 960, 3), dtype=np.uint8)
        auto = OpenVinoDetector({"model": str(MODEL_DIR), "device": "intel:auto"})
        cpu = OpenVinoDetector({"model": str(MODEL_DIR), "device": "intel:cpu"})
        auto.load(), cpu.load()
        assert auto.description.rsplit("@", 1)[-1] in ("CPU", "GPU")
        assert len(auto.infer(frame)) == len(cpu.infer(frame))


@pytest.mark.integration
class TestOpenVinoBackendWithTheRealModel:
    def test_wrong_imgsz_fails_at_startup_with_the_right_one_named(self):
        pytest.importorskip("openvino")
        if not MODEL_DIR.is_dir():
            pytest.skip("default model not present")
        with pytest.raises(ModelLoadError, match=r"\[544, 960\]"):
            OpenVinoDetector({"model": str(MODEL_DIR), "imgsz": [384, 640]}).load()

    def test_matches_the_ultralytics_backend_box_for_box(self):
        pytest.importorskip("openvino")
        pytest.importorskip("ultralytics")
        if not MODEL_DIR.is_dir():
            pytest.skip("default model not present")
        from computer_vision.inference.ultralytics_backend import UltralyticsDetector

        params = {"model": str(MODEL_DIR), "device": "intel:cpu", "imgsz": [544, 960]}
        direct, reference = OpenVinoDetector(params), UltralyticsDetector(params)
        direct.load()
        reference.load()

        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
        frame[100:400, 300:500] = (40, 80, 160)
        ours, theirs = direct.infer(frame), reference.infer(frame)
        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            assert a.label == b.label
            assert a.confidence == pytest.approx(b.confidence, abs=1e-3)
            assert a.xyxy == pytest.approx(b.xyxy, abs=0.5)


# --------------------------------------------------------------------------- #
# Live view
# --------------------------------------------------------------------------- #


class TestSourcePeek:
    def test_a_source_that_cannot_be_peeked_says_so(self):
        from computer_vision.sources.base import BaseVideoSource

        assert BaseVideoSource.peek(object()) == (None, 0)

    def test_peek_does_not_take_the_frame_from_the_pipeline(self, fake_stream):
        source = RTSPSource(rtsp_config())
        assert source.peek() == (None, 0)  # not open yet
        source.open()
        try:
            assert wait_for(lambda: source.peek()[0] is not None)
            frame, sequence = source.peek()
            again, same = source.peek()
            assert frame is not None and sequence > 0
            assert same >= sequence and again is not None
            assert source.read() is not None  # still there for the pipeline
        finally:
            source.release()
        assert source.peek() == (None, 0)

    def test_the_sequence_advances_with_new_frames(self, fake_stream):
        source = RTSPSource(rtsp_config())
        source.open()
        try:
            assert wait_for(lambda: source.peek()[0] is not None)
            first = source.peek()[1]
            assert wait_for(lambda: source.peek()[1] > first)
        finally:
            source.release()


class TestFramePrepare:
    def test_matches_what_analysis_sees(self):
        from computer_vision.sources.base import FramePacket

        processor = FrameProcessor(ProcessingConfig(target_width=320, mirror=True))
        raw_frame = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        packet = FramePacket(frame=raw_frame, frame_id=1, source_id="x")
        assert np.array_equal(processor.prepare(raw_frame), processor.build_context(packet).frame)


class TestDashboardStateLiveFrames:
    def test_a_live_frame_replaces_the_picture_only(self):
        state = DashboardState()
        state.publish(jpeg=b"analysed", summary={"frame_id": 7}, stats={"fps": 2}, events=[])
        state.publish_frame(b"live")

        assert state.latest_frame() == (b"live", 2)
        snapshot = state.snapshot()
        assert snapshot["frame"] == {"frame_id": 7} and snapshot["stats"] == {"fps": 2}

    def test_live_frames_do_not_hide_a_stalled_pipeline(self):
        state = DashboardState()
        state.publish_frame(b"live")
        assert state.seconds_since_publish() is None

    def test_viewer_demand(self):
        state = DashboardState()
        assert state.viewer_demand() == "idle"
        state.latest_jpeg()
        assert state.viewer_demand() == "snapshot"
        state.latest_frame()
        assert state.viewer_demand() == "stream"


class TestLiveView:
    def _sink(self, state, fps=50.0) -> WebSink:
        return WebSink(state, ProcessingConfig(target_width=None), stream_fps=fps)

    def _camera(self):
        """A peek that yields a new frame on every call."""
        counter = {"n": 0}

        def peek():
            counter["n"] += 1
            return np.full((48, 64, 3), counter["n"] % 255, dtype=np.uint8), counter["n"]

        return peek

    def test_without_a_live_view_every_picture_is_an_analysed_frame(self):
        state = DashboardState()
        sink = self._sink(state)
        sink.emit(context(), stats())
        assert not sink.live_view_active
        assert state.latest_frame()[1] == 1
        sink.close()

    def test_the_picture_keeps_moving_while_analysis_is_stuck(self):
        state = DashboardState()
        state.latest_frame()  # a viewer has the stream open
        sink = self._sink(state)
        sink.attach_live_view(self._camera(), lambda frame: frame)
        sink.emit(context(), stats())  # one analysed frame, then nothing more
        try:
            keep_watching = lambda: state.latest_frame()[1]  # noqa: E731
            assert wait_for(lambda: keep_watching() >= 10)
            assert sink.live_view_active
            assert state.snapshot()["frame"]["frame_id"] == 1  # results untouched
        finally:
            sink.close()

    def test_analysed_frames_are_not_also_encoded_while_live(self):
        state = DashboardState()
        state.latest_frame()
        sink = self._sink(state)
        sink.attach_live_view(self._camera(), lambda frame: frame)
        sink.emit(context(1), stats())
        try:
            assert wait_for(lambda: sink.live_view_active)
            encoded = []
            sink._encode = lambda ctx: encoded.append(ctx) or b"x"
            sink.emit(context(2), stats())
            assert encoded == []
            assert state.snapshot()["frame"]["frame_id"] == 2
        finally:
            sink.close()

    def test_events_are_published_exactly_once(self):
        from computer_vision.core.context import Event

        state = DashboardState()
        state.latest_frame()
        sink = self._sink(state)
        sink.attach_live_view(self._camera(), lambda frame: frame)
        ctx = context()
        ctx.add_event(
            Event(event_type="zone_entry", source_module="restricted_zone", frame_id=1,
                  timestamp="2026-01-01T00:00:00Z")
        )
        sink.emit(ctx, stats())
        try:
            assert wait_for(lambda: state.latest_frame()[1] >= 5)
            assert len(state.events_since(0)) == 1
        finally:
            sink.close()

    def test_falls_back_to_analysed_frames_when_the_source_cannot_be_peeked(self):
        state = DashboardState()
        sink = self._sink(state, fps=0)
        sink.attach_live_view(lambda: (None, 0), lambda frame: frame)
        try:
            for frame_id in (1, 2, 3):
                sink.emit(context(frame_id), stats())
            assert not sink.live_view_active
            assert state.latest_frame()[1] == 3
        finally:
            sink.close()

    def test_the_captures_own_frame_is_never_drawn_on(self):
        state = DashboardState()
        state.latest_frame()
        shared = np.zeros((48, 64, 3), dtype=np.uint8)
        calls = {"n": 0}

        def peek():
            calls["n"] += 1
            return shared, calls["n"]

        sink = self._sink(state)
        sink.attach_live_view(peek, lambda frame: frame)
        ctx = context()
        ctx.add_tracks(
            "person_tracking",
            [Track(track_id=1, label="person", bbox=BBox(5, 5, 40, 40), confidence=0.9)],
        )
        sink.emit(ctx, stats())
        try:
            assert wait_for(lambda: state.latest_frame()[1] >= 3)
            assert not shared.any()
        finally:
            sink.close()

    def test_old_boxes_are_not_drawn_on_a_live_frame(self, monkeypatch):
        from backend.services import web_sink

        drawn = []
        monkeypatch.setattr(web_sink, "draw_tracks", lambda frame, tracks: drawn.append(1))
        monkeypatch.setattr(web_sink, "STALE_RESULTS_SEC", 0.05)

        state = DashboardState()
        state.latest_frame()
        sink = self._sink(state)
        sink.attach_live_view(self._camera(), lambda frame: frame)
        sink.emit(context(), stats())
        try:
            time.sleep(0.2)
            drawn.clear()
            before = state.latest_frame()[1]
            assert wait_for(lambda: state.latest_frame()[1] >= before + 3)
            assert drawn == []
        finally:
            sink.close()

    def test_rate_follows_who_is_watching(self):
        state = DashboardState()
        sink = self._sink(state, fps=20.0)
        assert sink._live_interval() == pytest.approx(2.0)  # nobody
        state.latest_jpeg()
        assert sink._live_interval() == pytest.approx(0.5)  # a thumbnail
        state.latest_frame()
        assert sink._live_interval() == pytest.approx(0.05)  # a stream
        sink.close()

    def test_close_stops_the_thread(self):
        state = DashboardState()
        state.latest_frame()
        sink = self._sink(state)
        sink.attach_live_view(self._camera(), lambda frame: frame)
        sink.emit(context(), stats())
        assert wait_for(lambda: sink.live_view_active)
        thread = sink._live_thread
        sink.close()
        assert not thread.is_alive()
        assert not state.running

    def test_a_failing_frame_does_not_end_the_view(self):
        state = DashboardState()
        state.latest_frame()
        calls = {"n": 0}

        def prepare(frame):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("bad frame")
            return frame

        sink = self._sink(state)
        sink.attach_live_view(self._camera(), prepare)
        sink.emit(context(), stats())
        try:
            assert wait_for(lambda: state.latest_frame()[1] >= 3)
        finally:
            sink.close()


class TestLiveViewConfig:
    def test_on_by_default(self):
        assert DashboardConfig().live_view is True

    def test_can_be_turned_off(self):
        assert DashboardConfig(live_view=False).live_view is False


class TestStreamKeepAlive:
    def _collect(self, state, count, monkeypatch, keepalive):
        monkeypatch.setattr(server_module, "_STREAM_KEEPALIVE_SEC", keepalive)
        slots = server_module._StreamSlots(2)

        async def run():
            chunks = []
            generator = server_module._mjpeg_frames(state, slots, 0.01)
            try:
                async for chunk in generator:
                    chunks.append(chunk)
                    if len(chunks) >= count:
                        break
            finally:
                await generator.aclose()
            return chunks

        chunks = asyncio.run(asyncio.wait_for(run(), timeout=5))
        assert slots.active == 0
        return chunks

    def test_the_current_frame_is_repeated_while_nothing_new_arrives(self, monkeypatch):
        state = DashboardState()
        state.publish(jpeg=b"\xff\xd8one", summary={}, stats={}, events=[])
        chunks = self._collect(state, 3, monkeypatch, keepalive=0.05)
        assert len(chunks) == 3 and all(b"\xff\xd8one" in chunk for chunk in chunks)

    def test_nothing_is_sent_before_the_first_frame(self, monkeypatch):
        state = DashboardState()
        monkeypatch.setattr(server_module, "_STREAM_KEEPALIVE_SEC", 0.01)
        slots = server_module._StreamSlots(2)

        async def run():
            generator = server_module._mjpeg_frames(state, slots, 0.01)
            task = asyncio.ensure_future(generator.__anext__())
            await asyncio.sleep(0.1)
            assert not task.done()
            state.stop()
            with pytest.raises(StopAsyncIteration):
                await task

        asyncio.run(asyncio.wait_for(run(), timeout=5))


class TestLiveViewWiring:
    def _sink(self) -> WebSink:
        return WebSink(DashboardState(), ProcessingConfig(target_width=None))

    def _config(self):
        from computer_vision.core.config import VisionGuardConfig

        return VisionGuardConfig.model_validate({"processing": {"show_window": False}})

    def test_an_rtsp_source_gets_a_live_view(self, fake_stream):
        from backend.services.runtime import _attach_live_view

        sink = self._sink()
        _attach_live_view(sink, RTSPSource(rtsp_config()), self._config())
        assert sink._peek is not None

    def test_a_source_that_cannot_be_peeked_gets_none(self):
        from backend.services.runtime import _attach_live_view
        from computer_vision.sources.video_file import VideoFileSource

        sink = self._sink()
        _attach_live_view(sink, VideoFileSource.__new__(VideoFileSource), self._config())
        _attach_live_view(sink, object(), self._config())
        assert sink._peek is None

    def test_the_dashboard_sink_is_found_inside_a_multi_sink(self, fake_stream):
        from backend.services.runtime import _attach_live_view
        from computer_vision.core.output import MultiSink, NullSink

        web = self._sink()
        _attach_live_view(MultiSink([NullSink(), web]), RTSPSource(rtsp_config()), self._config())
        assert web._peek is not None


# --------------------------------------------------------------------------- #
# Tracker timestamps
# --------------------------------------------------------------------------- #


class TestTrackerTimestamps:
    def _walk(self, times, timed):
        """A 40 px wide walker at 40 px/s, seen at the given moments."""
        tracker = IoUTracker(iou_threshold=0.2, min_hits=1)
        ids = set()
        for moment in times:
            x = 40.0 * moment
            tracks = tracker.update(
                [det(x, 100, x + 40, 200)], timestamp=moment if timed else None
            )
            ids.update(t.track_id for t in tracks)
        return ids, tracker

    #: Short gaps, then one three times as long.
    UNEVEN = (0.0, 0.5, 1.0, 1.5, 3.0, 3.5)

    def test_an_uneven_gap_loses_the_track_when_frames_are_only_counted(self):
        ids, _ = self._walk(self.UNEVEN, timed=False)
        assert len(ids) > 1

    def test_with_timestamps_the_long_gap_is_predicted_correctly(self):
        ids, tracker = self._walk(self.UNEVEN, timed=True)
        assert ids == {1}
        assert tracker.timed

    def test_velocity_is_published_in_pixels_per_second(self):
        _, tracker = self._walk((0.0, 0.5, 1.0, 1.5), timed=True)
        state = tracker.active_tracks[0]
        velocity = state.as_track("person_tracking", tracker.timed).extra["velocity"]
        assert velocity[0] == pytest.approx(40.0, rel=0.2) and abs(velocity[1]) < 1.0

    def test_no_velocity_is_published_without_timestamps(self):
        _, tracker = self._walk((0.0, 0.5, 1.0), timed=False)
        assert not tracker.timed
        assert "velocity" not in tracker.active_tracks[0].as_track("x", tracker.timed).extra

    def test_timestamps_that_do_not_advance_count_as_frames(self):
        """Contexts built by hand often carry the same timestamp throughout."""
        counted = IoUTracker(iou_threshold=0.2, min_hits=1)
        stamped = IoUTracker(iou_threshold=0.2, min_hits=1)
        for x in (0, 10, 20, 30, 40):
            a = counted.update([det(x, 100, x + 40, 200)])
            b = stamped.update([det(x, 100, x + 40, 200)], timestamp=0.0)
        assert not stamped.timed
        assert a[0].velocity == b[0].velocity
        assert a[0].predicted_bbox() == b[0].predicted_bbox()

    def test_reset_forgets_the_clock(self):
        _, tracker = self._walk((0.0, 0.5, 1.0), timed=True)
        tracker.reset()
        assert not tracker.timed

    def test_the_module_passes_frame_times_through(self):
        module = PersonTrackingModule({"min_hits": 1})
        module.initialize()
        for index in range(4):
            ctx = context(index + 1, ts=index * 0.5)
            x = index * 10
            ctx.add_detections("person_detection", [det(x, 100, x + 40, 200)])
            module.process(ctx)
        velocity = ctx.get_tracks("person_tracking")[0].extra["velocity"]
        assert velocity[0] == pytest.approx(20.0, rel=0.3)


# --------------------------------------------------------------------------- #
# Live view: where the boxes go
# --------------------------------------------------------------------------- #


def a_track(track_id, x, velocity=None, module="person_tracking") -> Track:
    extra = {} if velocity is None else {"velocity": list(velocity)}
    return Track(
        track_id=track_id, label="person", bbox=BBox(x, 100, x + 40, 200),
        confidence=0.9, source_module=module, extra=extra,
    )


class TestBoxesBetweenTwoResults:
    def _results(self, tracks, ts):
        from backend.services.web_sink import _Results

        return _Results(tracks=tracks, ts=ts, at=ts)

    def test_a_track_seen_at_both_ends_is_placed_in_between(self):
        from backend.services.web_sink import tracks_between

        before, after = self._results([a_track(1, 0)], 0.0), self._results([a_track(1, 100)], 1.0)
        shown = tracks_between(before, after, 0.25)
        assert [t.bbox.x1 for t in shown] == [25.0]
        assert shown[0].bbox.x2 == 65.0 and shown[0].track_id == 1

    def test_the_ends_are_exact(self):
        from backend.services.web_sink import tracks_between

        before, after = self._results([a_track(1, 0)], 0.0), self._results([a_track(1, 100)], 1.0)
        assert tracks_between(before, after, 0.0)[0].bbox.x1 == 0.0
        assert tracks_between(before, after, 1.0)[0].bbox.x1 == 100.0

    def test_a_track_that_vanished_is_shown_for_the_first_half_only(self):
        from backend.services.web_sink import tracks_between

        before, after = self._results([a_track(1, 0)], 0.0), self._results([], 1.0)
        assert len(tracks_between(before, after, 0.4)) == 1
        assert tracks_between(before, after, 0.6) == []

    def test_a_track_that_appeared_is_shown_for_the_second_half_only(self):
        from backend.services.web_sink import tracks_between

        before, after = self._results([], 0.0), self._results([a_track(7, 50)], 1.0)
        assert tracks_between(before, after, 0.4) == []
        assert [t.track_id for t in tracks_between(before, after, 0.6)] == [7]

    def test_ids_from_different_trackers_are_not_confused(self):
        from backend.services.web_sink import tracks_between

        before = self._results([a_track(1, 0, module="person_tracking")], 0.0)
        after = self._results([a_track(1, 500, module="vehicle_tracking")], 1.0)
        shown = tracks_between(before, after, 0.25)
        assert [t.bbox.x1 for t in shown] == [0.0]  # not dragged towards the vehicle

    def test_the_original_tracks_are_not_modified(self):
        from backend.services.web_sink import tracks_between

        first = a_track(1, 0)
        tracks_between(self._results([first], 0.0), self._results([a_track(1, 100)], 1.0), 0.5)
        assert first.bbox.x1 == 0.0


class TestBoxesMovedByVelocity:
    def test_a_box_moves_along_its_velocity(self):
        from backend.services.web_sink import tracks_moved_on

        moved = tracks_moved_on([a_track(1, 0, velocity=(20, 0))], 0.5)
        assert moved[0].bbox.x1 == pytest.approx(10.0) and moved[0].bbox.y1 == 100

    def test_never_further_than_its_own_size(self):
        from backend.services.web_sink import tracks_moved_on

        moved = tracks_moved_on([a_track(1, 0, velocity=(5000, 0))], 1.0)
        assert moved[0].bbox.x1 == pytest.approx(100.0)  # the box is 100 px tall

    def test_time_is_capped(self):
        from backend.services.web_sink import tracks_moved_on

        assert tracks_moved_on([a_track(1, 0, velocity=(10, 0))], 60.0)[0].bbox.x1 == 15.0

    def test_a_track_without_velocity_stays_where_it_was(self):
        from backend.services.web_sink import tracks_moved_on

        track = a_track(1, 0)
        assert tracks_moved_on([track], 1.0) == [track]


class TestSyncedView:
    def _sink(self, results):
        sink = WebSink(DashboardState(), ProcessingConfig(target_width=None))
        sink._history = tuple(results)
        return sink

    def _results(self, x, ts):
        from backend.services.web_sink import _Results

        return _Results(tracks=[a_track(1, x)], ts=ts, at=ts)

    def _frame(self, value):
        return np.full((4, 4, 3), value, dtype=np.uint8)

    def test_needs_two_results(self):
        sink = self._sink([self._results(0, 10.0)])
        assert sink._synced_frame(self._frame(1), 10.1) is None

    def test_gives_up_when_analysis_is_too_far_behind(self):
        sink = self._sink([self._results(0, 10.0), self._results(100, 11.0)])
        assert sink._synced_frame(self._frame(1), 20.0) is None

    def test_plays_the_interval_between_two_results_with_boxes_in_step(self):
        sink = self._sink([self._results(0, 10.0), self._results(100, 11.0)])
        # Frames captured through the interval, before the second result
        # arrived (at 11.4) made them showable.
        for index, captured in enumerate((10.0, 10.25, 10.5, 10.75, 11.0, 11.25)):
            sink._frames.append((captured, self._frame(index)))

        positions = []
        for step in range(5):  # shown from 11.4 on, a quarter second apart
            picture, _, tracks = sink._synced_frame(self._frame(99), 11.4 + step * 0.25)
            positions.append((int(picture[0, 0, 0]), round(tracks[0].bbox.x1)))

        # frame index, box position: the box is where the walker was on
        # each frame, not where the last result left it.
        assert positions == [(0, 0), (1, 25), (2, 50), (3, 75), (4, 100)]

    def test_never_runs_ahead_of_what_was_analysed(self):
        sink = self._sink([self._results(0, 10.0), self._results(100, 11.0)])
        sink._frames.append((10.0, self._frame(0)))
        sink._frames.append((11.0, self._frame(1)))
        sink._frames.append((11.5, self._frame(2)))
        sink._synced_frame(self._frame(3), 11.4)
        for step in range(1, 6):  # two and a half seconds with no new result
            sink._synced_frame(self._frame(3), 11.4 + step * 0.5)
        assert sink._playhead == 11.0

    def test_waits_without_republishing_the_same_frame(self):
        sink = self._sink([self._results(0, 10.0), self._results(100, 11.0)])
        sink._frames.append((10.0, self._frame(0)))
        first = sink._synced_frame(self._frame(1), 10.01)
        again = sink._synced_frame(self._frame(1), 10.01)  # no time has passed
        assert first[0] is not None and again[0] is None

    def test_catches_up_when_it_has_fallen_behind(self):
        sink = self._sink([self._results(0, 10.0), self._results(100, 11.0)])
        sink._frames.append((10.0, self._frame(0)))
        sink._synced_frame(self._frame(1), 11.4)
        sink._history = tuple(self._results(x * 100, 10.0 + x) for x in range(2, 6))  # 12..15
        sink._playhead_moved_at = 15.1  # it has not been advancing meanwhile
        sink._synced_frame(self._frame(1), 15.2)
        assert 12.0 <= sink._playhead <= 15.0  # within reach of the newest result again

    def test_runs_further_behind_when_there_is_history_to_absorb_a_late_result(self):
        sink = self._sink([self._results(x * 100, 10.0 + x) for x in range(4)])  # 10..13
        sink._frames.append((11.0, self._frame(0)))
        sink._synced_frame(self._frame(1), 13.3)
        assert sink._playhead == pytest.approx(11.5)

    def test_a_viewer_who_just_connected_sees_the_newest_frame_first(self):
        sink = self._sink([self._results(0, 10.0), self._results(100, 11.0)])
        # Nothing was being kept until now, 1.4 s after the playhead's moment.
        assert sink._synced_frame(self._frame(1), 11.4) is None
        assert sink._playhead is not None  # still advancing, to take over shortly

    def test_old_frames_are_dropped(self):
        sink = self._sink([])
        for index in range(200):
            sink._synced_frame(self._frame(1), index * 0.1)
        assert len(sink._frames) <= 45

    def test_the_synced_view_can_be_turned_off(self):
        from computer_vision.core.config import DashboardConfig as Config

        assert Config().live_view_sync is True
        sink = WebSink(DashboardState(), ProcessingConfig(target_width=None), sync=False)
        sink._history = (self._results(0, 10.0), self._results(100, 11.0))
        sink._state.latest_frame()
        sink._peek = lambda: (None, 0)
        sink._prepare = lambda frame: frame
        assert sink._encode_live(self._frame(1)) is not None
        assert len(sink._frames) == 0

    def test_end_to_end_a_delayed_picture_is_published(self):
        state = DashboardState()
        state.latest_frame()
        counter = {"n": 0}

        def peek():
            counter["n"] += 1
            return np.full((48, 64, 3), counter["n"] % 255, dtype=np.uint8), counter["n"]

        sink = WebSink(state, ProcessingConfig(target_width=None), stream_fps=50.0)
        sink.attach_live_view(peek, lambda frame: frame)
        try:
            for frame_id in range(1, 6):  # analysis at about 10 fps
                ctx = context(frame_id, ts=time.perf_counter())
                ctx.add_tracks("person_tracking", [a_track(1, frame_id * 5)])
                sink.emit(ctx, stats())
                time.sleep(0.1)
            assert sink._playhead is not None  # the synced path is in use
            assert state.latest_frame()[1] >= 5
        finally:
            sink.close()
