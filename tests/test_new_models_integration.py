"""Real-model checks for number-plate reading and person re-identification.

Deselected by default (``-m integration``): they need the ``openvino`` package
and the model files under ``computer_vision/Model/``. Each test skips itself
when its model is absent, so a checkout without the optional 107 MB
re-identification model still passes the rest.

``assets/plate_sample.png`` is the sample image from the MIT-licensed
``fast-alpr`` project; its plate reads ``5AU5341``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("openvino")

ROOT = Path(__file__).resolve().parents[1]
ANPR_DETECTOR = ROOT / "computer_vision/Model/anpr/plate_detector_yolov9t_384.onnx"
ANPR_OCR = ROOT / "computer_vision/Model/anpr/plate_ocr_cct_s_v2_global.onnx"
REID_MODEL = ROOT / "computer_vision/Model/reid/person_reid_youtu_2021nov.onnx"
SAMPLE = Path(__file__).parent / "assets" / "plate_sample.png"

DETECTOR_SHA = "888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8"
OCR_SHA = "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b"


@pytest.fixture(scope="module")
def reader():
    if not (ANPR_DETECTOR.is_file() and ANPR_OCR.is_file()):
        pytest.skip("ANPR models not present")
    from computer_vision.inference.plate_reader import PlateReader

    return PlateReader(
        str(ANPR_DETECTOR), str(ANPR_OCR), detector_sha256=DETECTOR_SHA, ocr_sha256=OCR_SHA
    )


class TestPlateReader:
    def test_finds_and_reads_the_sample_plate(self, reader):
        image = cv2.imread(str(SAMPLE))
        plates = reader.detect(image)

        assert len(plates) == 1
        plate = plates[0]
        # Reference values from fast-alpr on the same image.
        assert (plate.x1, plate.y1, plate.x2, plate.y2) == (307, 187, 387, 210)
        assert plate.confidence == pytest.approx(0.843, abs=0.01)

        read = reader.read(image[plate.y1:plate.y2, plate.x1:plate.x2])
        assert read.text == "5AU5341"
        assert read.confidence > 0.95

    def test_a_blank_image_has_no_plates(self, reader):
        assert reader.detect(np.full((300, 400, 3), 127, dtype=np.uint8)) == []

    def test_wrong_hash_refuses_to_load(self):
        if not ANPR_DETECTOR.is_file():
            pytest.skip("ANPR models not present")
        from computer_vision.inference.base import ModelLoadError
        from computer_vision.inference.plate_reader import PlateReader

        with pytest.raises(ModelLoadError, match="integrity"):
            PlateReader(str(ANPR_DETECTOR), str(ANPR_OCR), detector_sha256="0" * 64)


class TestNumberPlateModuleEndToEnd:
    def test_reports_the_plate_of_a_tracked_vehicle(self, reader):
        """The whole chain on a real image: the module gets a vehicle track,
        crops it from the original capture, finds the plate and votes."""
        from computer_vision.core.context import BBox, FrameContext, Track
        from computer_vision.modules.recognition.number_plate import NumberPlateRecognitionModule

        original = cv2.imread(str(SAMPLE))
        # The pipeline would hand modules a resized frame; mimic that so the
        # coordinate mapping between the two is exercised for real.
        frame = cv2.resize(original, (original.shape[1] // 2, original.shape[0] // 2))

        module = NumberPlateRecognitionModule({"interval_frames": 1, "votes": 2})
        module.initialize()

        events = []
        for frame_id in range(3):
            context = FrameContext(
                frame=frame, frame_id=frame_id, source_id="test",
                monotonic_ts=float(frame_id), original_frame=original,
            )
            vehicle = BBox(0, 0, frame.shape[1], frame.shape[0])  # the car fills the image
            context.add_tracks(
                "vehicle_tracking",
                [Track(track_id=1, label="car", bbox=vehicle, confidence=0.9, source_module="vehicle_tracking")],
            )
            module.process(context)
            events.extend(context.events)

        assert [e.data["plate"] for e in events] == ["5AU5341"]
        plate = context.get_detections("number_plate_recognition")[0]
        assert plate.label == "5AU5341"
        # In frame (half-size) coordinates, the plate sits where it is in the image.
        assert plate.bbox.x1 == pytest.approx(307 / 2, abs=2)
        assert plate.bbox.y1 == pytest.approx(187 / 2, abs=2)


class TestPersonEmbedder:
    def test_same_crop_matches_itself_and_not_noise(self):
        if not REID_MODEL.is_file():
            pytest.skip("re-identification model not present (optional, 107 MB)")
        from computer_vision.inference.person_embedder import PersonEmbedder

        embedder = PersonEmbedder(str(REID_MODEL))
        rng = np.random.default_rng(0)
        person = rng.integers(0, 255, (200, 80, 3), dtype=np.uint8)
        other = rng.integers(0, 255, (200, 80, 3), dtype=np.uint8)

        a, b, c = embedder.embed(person), embedder.embed(person.copy()), embedder.embed(other)
        assert a.shape == (768,)
        assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-4)
        assert float(a @ b) == pytest.approx(1.0, abs=1e-4)
        assert float(a @ c) < float(a @ b)
