"""Accuracy features: low-confidence tracker candidates, cross-label vehicle
tracking, and exported (OpenVINO) model support.

Each of these was adopted because it measurably improved tracking on real
footage; these tests pin the behaviour so a refactor cannot quietly lose it.
"""

from __future__ import annotations

import numpy as np
import pytest

from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference import ultralytics_backend
from computer_vision.inference.base import (
    BackendNotAvailableError,
    ModelLoadError,
    RawDetection,
    model_digest,
    verify_model_file,
)
from computer_vision.inference.factory import reset_detector_cache
from computer_vision.inference.shared import share_key
from computer_vision.inference.ultralytics_backend import parse_imgsz
from computer_vision.modules.base import ModuleInitializationError
from computer_vision.modules.detection.person import PersonDetectionModule
from computer_vision.modules.tracking.person import PersonTrackingModule
from computer_vision.modules.tracking.tracker_core import IoUTracker
from computer_vision.modules.tracking.vehicle import VehicleTrackingModule


def det(label: str = "person", conf: float = 0.9, box=(100, 100, 150, 250)) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(*box))


def raw(label: str, conf: float, box=(100, 100, 150, 250)) -> RawDetection:
    return RawDetection(label=label, confidence=conf, xyxy=box, class_id=-1)


def frame_ctx(frame_id: int = 1) -> FrameContext:
    return FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        source_id="test",
        monotonic_ts=0.0,
    )


@pytest.fixture(autouse=True)
def _fresh_detectors():
    reset_detector_cache()
    yield
    reset_detector_cache()


# --------------------------------------------------------------------------- #
# Tracker: low-confidence candidates
# --------------------------------------------------------------------------- #


class TestTrackerCandidates:
    def _confirmed(self, **kwargs) -> IoUTracker:
        tracker = IoUTracker(min_hits=2, **kwargs)
        tracker.update([det()])
        assert [t.track_id for t in tracker.update([det()])] == [1]
        return tracker

    def test_a_candidate_continues_a_confirmed_track(self):
        tracker = self._confirmed()
        reported = tracker.update([], candidates=[det(conf=0.3)])
        assert [t.track_id for t in reported] == [1]
        assert reported[0].time_since_update == 0
        assert reported[0].confidence == pytest.approx(0.3)

    def test_without_the_candidate_the_track_goes_unreported(self):
        tracker = self._confirmed()
        assert tracker.update([]) == []

    def test_a_candidate_never_starts_a_track(self):
        tracker = IoUTracker(min_hits=1)
        for _ in range(5):
            assert tracker.update([], candidates=[det(conf=0.3)]) == []
        assert tracker.total_ids_issued == 0

    def test_a_candidate_cannot_confirm_an_unconfirmed_track(self):
        tracker = IoUTracker(min_hits=2)
        tracker.update([det()])  # one hit: not confirmed yet
        assert tracker.update([], candidates=[det(conf=0.3)]) == []
        assert tracker.active_tracks[0].hits == 1

    def test_a_real_detection_wins_over_a_candidate(self):
        tracker = self._confirmed()
        reported = tracker.update([det(conf=0.8)], candidates=[det(conf=0.3)])
        assert reported[0].confidence == pytest.approx(0.8)

    def test_candidate_needs_enough_overlap(self):
        tracker = self._confirmed(candidate_iou_threshold=0.5)
        far = det(conf=0.3, box=(140, 100, 190, 250))  # IoU ~0.11 with the track
        assert tracker.update([], candidates=[far]) == []

    def test_candidate_of_another_label_is_ignored(self):
        tracker = self._confirmed()
        assert tracker.update([], candidates=[det(label="dog", conf=0.3)]) == []

    def test_one_candidate_serves_one_track(self):
        tracker = IoUTracker(min_hits=1)
        first, second = det(box=(100, 100, 150, 250)), det(box=(110, 100, 160, 250))
        tracker.update([first, second])
        reported = tracker.update([], candidates=[det(conf=0.3, box=(100, 100, 150, 250))])
        assert len(reported) == 1

    def test_bad_candidate_iou_is_rejected(self):
        with pytest.raises(ValueError):
            IoUTracker(candidate_iou_threshold=0.0)


class TestMatchAcrossLabels:
    def test_off_by_default_a_label_flip_starts_a_new_id(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det("car")])
        assert [t.track_id for t in tracker.update([det("truck")])] == [2]

    def test_on_a_label_flip_keeps_the_id_and_takes_the_new_label(self):
        tracker = IoUTracker(min_hits=1, match_across_labels=True)
        tracker.update([det("car")])
        reported = tracker.update([det("truck")])
        assert [(t.track_id, t.label) for t in reported] == [(1, "truck")]
        assert tracker.total_ids_issued == 1


# --------------------------------------------------------------------------- #
# Modules: detector publishes candidates, tracker consumes them
# --------------------------------------------------------------------------- #


class TestCandidatePlumbing:
    def _detector(self, **params) -> PersonDetectionModule:
        module = PersonDetectionModule({"backend": "fake", "confidence_threshold": 0.4, **params})
        module.initialize()
        return module

    def test_off_by_default(self, fake_detector_backend):
        fake_detector_backend.scripted = [raw("person", 0.9), raw("person", 0.3)]
        ctx = frame_ctx()
        self._detector().process(ctx)
        assert len(ctx.get_detections("person_detection")) == 1
        assert ctx.get_candidates("person_detection") == []

    def test_band_between_the_two_thresholds_becomes_candidates(self, fake_detector_backend):
        fake_detector_backend.scripted = [
            raw("person", 0.9),
            raw("person", 0.3),
            raw("person", 0.1),  # below candidate_threshold: dropped entirely
            raw("car", 0.3),  # wrong label: never a person candidate
        ]
        ctx = frame_ctx()
        self._detector(candidate_threshold=0.25).process(ctx)

        assert [d.confidence for d in ctx.get_detections("person_detection")] == [0.9]
        assert [d.confidence for d in ctx.get_candidates("person_detection")] == [0.3]

    def test_candidates_are_not_results(self, fake_detector_backend):
        fake_detector_backend.scripted = [raw("person", 0.3)]
        ctx = frame_ctx()
        result = self._detector(candidate_threshold=0.25).process(ctx)

        assert result.detections == []
        assert ctx.summary()["detections"]["person_detection"] == []
        assert "candidates" not in ctx.summary()

    def test_threshold_must_be_below_the_reporting_threshold(self):
        module = PersonDetectionModule(
            {"backend": "fake", "confidence_threshold": 0.4, "candidate_threshold": 0.4}
        )
        with pytest.raises(ModuleInitializationError):
            module.initialize()

    def test_tracker_keeps_an_id_through_a_weak_frame(self, fake_detector_backend):
        fake_detector_backend.scripted_sequence = [
            [raw("person", 0.9)],
            [raw("person", 0.9)],
            [raw("person", 0.3)],  # weak frame
            [raw("person", 0.9)],
        ]
        detector = self._detector(candidate_threshold=0.25)
        tracker = PersonTrackingModule({})
        tracker.initialize()

        ids = []
        for frame_id in range(4):
            ctx = frame_ctx(frame_id)
            detector.process(ctx)
            tracker.process(ctx)
            ids.append([t.track_id for t in ctx.get_tracks("person_tracking")])

        assert ids == [[], [1], [1], [1]]

    def test_vehicle_tracker_passes_its_options_through(self):
        module = VehicleTrackingModule(
            {"match_across_labels": True, "candidate_iou_threshold": 0.5}
        )
        module.initialize()
        assert module._tracker.match_across_labels is True
        assert module._tracker.candidate_iou_threshold == 0.5

    def test_tracker_defaults_are_unchanged(self):
        module = PersonTrackingModule({"iou_threshold": 0.2})
        module.initialize()
        assert module._tracker.match_across_labels is False
        assert module._tracker.candidate_iou_threshold == 0.2


# --------------------------------------------------------------------------- #
# Exported model support
# --------------------------------------------------------------------------- #


class TestImageSize:
    def test_single_number(self):
        assert parse_imgsz(960) == 960
        assert parse_imgsz("640") == 640
        assert parse_imgsz(None) == 640

    def test_height_width_pair(self):
        assert parse_imgsz([544, 960]) == (544, 960)

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ModelLoadError):
            parse_imgsz([1, 2, 3])

    def test_pair_shares_whether_list_or_tuple(self):
        a = share_key("ultralytics", {"model": "m", "imgsz": [544, 960]})
        b = share_key("ultralytics", {"model": "m", "imgsz": (544, 960)})
        c = share_key("ultralytics", {"model": "m", "imgsz": 960})
        assert a == b
        assert a != c


class TestModelDirectory:
    def _model_dir(self, tmp_path):
        folder = tmp_path / "net_openvino_model"
        folder.mkdir()
        (folder / "net.xml").write_text("<net/>", encoding="utf-8")
        (folder / "net.bin").write_bytes(b"\x00\x01\x02")
        return folder

    def test_directory_is_accepted_and_returned(self, tmp_path):
        folder = self._model_dir(tmp_path)
        assert verify_model_file(str(folder)) == str(folder)

    def test_matching_digest_passes(self, tmp_path):
        folder = self._model_dir(tmp_path)
        digest = model_digest(folder)
        assert verify_model_file(str(folder), expected_sha256=digest.upper()) == str(folder)

    @pytest.mark.parametrize("change", ["edit", "add", "remove", "rename"])
    def test_any_change_to_the_directory_fails_the_check(self, tmp_path, change):
        folder = self._model_dir(tmp_path)
        digest = model_digest(folder)

        if change == "edit":
            (folder / "net.bin").write_bytes(b"\x00\x01\x03")
        elif change == "add":
            (folder / "extra.bin").write_bytes(b"x")
        elif change == "remove":
            (folder / "net.xml").unlink()
        else:
            (folder / "net.xml").rename(folder / "other.xml")

        with pytest.raises(ModelLoadError, match="integrity"):
            verify_model_file(str(folder), expected_sha256=digest)

    def test_empty_directory_is_an_error(self, tmp_path):
        folder = tmp_path / "empty_openvino_model"
        folder.mkdir()
        with pytest.raises(ModelLoadError, match="empty"):
            verify_model_file(str(folder))

    def test_file_digest_is_the_plain_sha256(self, tmp_path):
        import hashlib

        weights = tmp_path / "m.pt"
        weights.write_bytes(b"weights")
        assert model_digest(weights) == hashlib.sha256(b"weights").hexdigest()

    def test_missing_openvino_gives_install_instructions(self, monkeypatch):
        monkeypatch.setattr(ultralytics_backend.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(BackendNotAvailableError, match="pip install openvino"):
            ultralytics_backend._require_openvino("x_openvino_model")
