"""Small / hazy CCTV faces: head-crop detection and jitter-tolerant face tracking."""

from __future__ import annotations

import numpy as np

from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference.face_base import FaceDetectorBackend, RawFace
from computer_vision.modules.recognition.face_detection import FaceDetectionModule, _plausible_face
from computer_vision.modules.recognition.face_tracker import FaceTracker

FRAME_W, FRAME_H = 960, 540


def _face_in(width: int, height: int, size: float = 40.0, confidence: float = 0.5) -> RawFace:
    """A well-formed face centred in an image of the given size."""
    cx, cy, r = width / 2, height / 2, size / 2
    landmarks = np.array(
        [[cx - 0.25 * size, cy - 0.12 * size], [cx + 0.25 * size, cy - 0.12 * size],
         [cx, cy + 0.08 * size], [cx - 0.2 * size, cy + 0.28 * size], [cx + 0.2 * size, cy + 0.28 * size]],
        np.float32,
    )
    row = np.concatenate([[cx - r, cy - r, size, size], landmarks.reshape(-1), [confidence]]).astype(np.float32)
    return RawFace(confidence=confidence, xyxy=(cx - r, cy - r, cx + r, cy + r), landmarks=landmarks, raw_row=row)


class _CropOnlyBackend(FaceDetectorBackend):
    """Sees a face only in an enlarged crop - never in the whole frame."""

    name = "fake"

    def __init__(self) -> None:
        super().__init__({})
        self.shapes: list[tuple[int, int]] = []

    def load(self) -> None:
        self._loaded = True

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self.shapes.append((w, h))
        return [] if (w, h) == (FRAME_W, FRAME_H) else [_face_in(w, h)]


def _context(person=(300, 100, 360, 300), with_original=True) -> FrameContext:
    frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
    original = np.full((FRAME_H * 2, FRAME_W * 2, 3), 90, np.uint8) if with_original else None
    ctx = FrameContext(frame=frame, frame_id=1, source_id="t", monotonic_ts=0.0, original_frame=original)
    ctx.add_detections("person_detection", [Detection(label="person", confidence=0.9, bbox=BBox(*person))])
    return ctx


def _module(**params) -> tuple[FaceDetectionModule, _CropOnlyBackend]:
    module = FaceDetectionModule({"motion_threshold": 0, "head_crops": True, "crop_confirm_hits": 1, **params})
    backend = _CropOnlyBackend()
    module._backend = backend
    return module, backend


# ------------------------------------------------------------ head crops ---


def test_a_face_the_whole_frame_pass_misses_is_found_in_the_head_crop():
    module, backend = _module()
    ctx = _context()
    module.process(ctx)
    faces = ctx.get_detections("face_detection")
    assert len(faces) == 1
    # The crop is a head-sized, enlarged image - not the frame.
    assert any(shape != (FRAME_W, FRAME_H) for shape in backend.shapes)
    # Result is back in frame coordinates, inside the person's head region.
    box = faces[0].bbox
    assert 300 - 9 <= box.x1 and box.x2 <= 360 + 9
    assert 100 - 16 <= box.y1 and box.y2 <= 100 + 0.45 * 200 + 1


def test_crop_landmarks_and_alignment_row_are_in_frame_coordinates():
    module, _ = _module()
    ctx = _context()
    module.process(ctx)
    face = ctx.get_detections("face_detection")[0].extra["face"]
    x1, y1, x2, y2 = face.xyxy
    assert np.all((face.landmarks[:, 0] >= x1) & (face.landmarks[:, 0] <= x2))
    assert np.all((face.landmarks[:, 1] >= y1) & (face.landmarks[:, 1] <= y2))
    # recognition aligns with raw_row on the frame, so it must agree with xyxy
    assert np.allclose(face.raw_row[0:4], (x1, y1, x2 - x1, y2 - y1), atol=1e-3)
    assert np.allclose(face.raw_row[4:14], face.landmarks.reshape(-1), atol=1e-3)


def test_crops_also_work_when_the_frame_was_not_resized():
    module, _ = _module()
    ctx = _context(with_original=False)
    module.process(ctx)
    assert len(ctx.get_detections("face_detection")) == 1


def test_head_crops_off_keeps_the_plain_behaviour():
    module, backend = _module(head_crops=False)
    ctx = _context()
    module.process(ctx)
    assert ctx.get_detections("face_detection") == []
    assert backend.shapes == [(FRAME_W, FRAME_H)]


def test_no_person_means_no_crops_but_the_plain_pass_still_runs():
    module, backend = _module()
    ctx = _context()
    ctx.detections.clear()
    module.process(ctx)
    assert backend.shapes == [(FRAME_W, FRAME_H)]


def test_a_texture_that_is_not_a_face_is_rejected_by_the_geometry_check():
    face = _face_in(100, 100, size=40)
    assert _plausible_face(face.xyxy, face.landmarks)
    upside_down = face.landmarks[::-1].copy()          # mouth above the eyes
    assert not _plausible_face(face.xyxy, upside_down)
    assert not _plausible_face((0, 0, 80, 20), face.landmarks)   # not square-ish


def test_a_crop_face_needs_to_be_confirmed_by_a_second_detection():
    module, _ = _module(crop_confirm_hits=2)
    first, second = _context(), _context()
    module.process(first)
    second.monotonic_ts = 0.1
    module.process(second)
    assert first.get_detections("face_detection") == []
    assert len(second.get_detections("face_detection")) == 1


def test_a_face_shaped_texture_that_never_moves_is_dropped_after_a_while():
    module, _ = _module(crop_confirm_hits=1, crop_static_seconds=3.0)
    counts = []
    for n in range(8):                      # same spot, every 0.8 s
        ctx = _context()
        ctx.monotonic_ts = n * 0.8
        module.process(ctx)
        counts.append(len(ctx.get_detections("face_detection")))
    assert counts[:4] == [1, 1, 1, 1] and counts[-1] == 0


# --------------------------------------------------------- face tracking ---


def test_a_tiny_jittering_face_keeps_one_track_id_with_the_centre_gate():
    def run(gate):
        tracker = FaceTracker(center_gate=gate)
        ids = []
        for frame, dx in enumerate([0, 4, -4, 4, -4, 4, 0]):    # 12 px face, 8 px jitter swings
            [tid], _ = tracker.update([(100 + dx, 100, 112 + dx, 112)], frame, frame * 0.04)
            ids.append(tid)
        return len(set(ids))

    assert run(0.0) > 1          # IoU alone fragments it
    assert run(0.8) == 1


def test_centre_gate_does_not_merge_faces_that_are_far_apart():
    tracker = FaceTracker(center_gate=0.8)
    (a, b), _ = tracker.update([(100, 100, 112, 112), (200, 100, 212, 112)], 0, 0.0)
    (a2, b2), _ = tracker.update([(103, 100, 115, 112), (203, 100, 215, 112)], 1, 0.04)
    assert (a2, b2) == (a, b) and a != b


def test_coasting_prediction_is_capped():
    tracker = FaceTracker()
    for frame in range(3):
        tracker.update([(10 * frame, 0, 10 * frame + 40, 40)], frame, frame * 0.04)
    track = next(iter(tracker.active.values()))
    assert np.allclose(track.predicted(2 + 100), track.predicted(2 + 8))


# ------------------------------------------------- YOLO-face + confirmation ---


class _FixedBackend(FaceDetectorBackend):
    name = "fixed"

    def __init__(self, faces) -> None:
        super().__init__({})
        self.faces = faces

    def load(self) -> None:
        self._loaded = True

    def detect(self, frame: np.ndarray):
        return list(self.faces)


def _face_at(cx: float, cy: float, size: float, confidence: float) -> RawFace:
    face = _face_in(int(2 * cx), int(2 * cy), size=size, confidence=confidence)
    return face


def _yolo_style_module(faces) -> FaceDetectionModule:
    module = FaceDetectionModule({
        "motion_threshold": 0, "confidence_threshold": 0.4, "min_face_size": 10,
        "confirm_low_confidence": True, "trusted_confidence": 0.6, "trusted_min_face_size": 24,
    })
    module._backend = _FixedBackend(faces)
    return module


def _tick(module, n):
    ctx = _context()
    ctx.monotonic_ts = n * 0.1
    module.process(ctx)
    return ctx.get_detections("face_detection")


def test_a_strong_full_size_face_is_published_at_once():
    module = _yolo_style_module([_face_at(200, 150, 40, 0.85)])
    assert len(_tick(module, 0)) == 1


def test_a_weak_small_face_needs_a_second_detection():
    module = _yolo_style_module([_face_at(200, 150, 16, 0.45)])
    assert _tick(module, 0) == []
    assert len(_tick(module, 1)) == 1


def test_a_weak_face_below_the_detector_threshold_is_never_published():
    module = _yolo_style_module([_face_at(200, 150, 40, 0.30)])
    assert all(_tick(module, n) == [] for n in range(4))


def test_yoloface_backend_is_registered_and_names_the_missing_model():
    import pytest

    from computer_vision.inference.face_base import ModelLoadError
    from computer_vision.inference.face_factory import available_face_detectors, create_face_detector

    assert "yoloface" in available_face_detectors()
    with pytest.raises(ModelLoadError, match="not found"):
        create_face_detector("yoloface", {"model": "nowhere/yolov8n-face.pt"}).load()
