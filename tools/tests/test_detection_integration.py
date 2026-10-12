"""End-to-end detection against a real model and a real image.

Skipped automatically when ultralytics is not installed or the model weights
are not present, so the core suite stays fast and hardware-free. Run it
explicitly with:

    pytest tools/tests/test_detection_integration.py -m integration
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from computer_vision.core.context import FrameContext
from computer_vision.modules.detection.person import PersonDetectionModule

pytestmark = pytest.mark.integration

ultralytics = pytest.importorskip(
    "ultralytics", reason="ultralytics not installed - detection backend unavailable"
)


def _bundled_asset(name: str) -> Path:
    """Ultralytics ships test images inside the package."""
    return Path(ultralytics.__file__).parent / "assets" / name


@pytest.fixture(scope="module")
def detector():
    module = PersonDetectionModule(
        {"model": "computer_vision/Model/detection/yolo11n.pt", "device": "cpu", "confidence_threshold": 0.4}
    )
    try:
        module.initialize()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load the detection model: {exc}")
    yield module
    module.cleanup()


def _detect(detector, image_path: Path) -> FrameContext:
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"could not read {image_path}"
    context = FrameContext(
        frame=frame, frame_id=1, source_id="integration", monotonic_ts=0.0
    )
    detector.process(context)
    return context


def test_finds_the_people_in_bus_jpg(detector):
    asset = _bundled_asset("bus.jpg")
    if not asset.exists():
        pytest.skip("bundled asset bus.jpg not found")

    context = _detect(detector, asset)
    detections = context.get_detections("person_detection")

    # The image contains four clearly visible people.
    assert len(detections) >= 3, f"expected >=3 people, got {len(detections)}"
    assert all(det.label == "person" for det in detections)
    assert all(0.4 <= det.confidence <= 1.0 for det in detections)


def test_boxes_are_inside_the_frame(detector):
    asset = _bundled_asset("bus.jpg")
    if not asset.exists():
        pytest.skip("bundled asset bus.jpg not found")

    context = _detect(detector, asset)

    for det in context.get_detections("person_detection"):
        x1, y1, x2, y2 = det.bbox.as_tuple()
        assert 0 <= x1 < x2 <= context.width
        assert 0 <= y1 < y2 <= context.height
        assert det.bbox.area > 0


def test_vehicles_are_not_reported(detector):
    """bus.jpg contains a large bus; person detection must ignore it."""
    asset = _bundled_asset("bus.jpg")
    if not asset.exists():
        pytest.skip("bundled asset bus.jpg not found")

    context = _detect(detector, asset)
    labels = {det.label for det in context.get_detections("person_detection")}

    assert labels <= {"person"}


def test_result_payload_is_json_safe(detector):
    import json

    asset = _bundled_asset("zidane.jpg")
    if not asset.exists():
        pytest.skip("bundled asset zidane.jpg not found")

    frame = cv2.imread(str(asset))
    context = FrameContext(
        frame=frame, frame_id=7, source_id="integration", monotonic_ts=0.0
    )
    result = detector.process(context)

    json.dumps(result.to_dict())
    json.dumps(context.summary())
    assert result.module == "person_detection"
