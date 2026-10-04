"""Number-plate reading: find the plate on a vehicle, then read its text.

Two small ONNX networks, run with OpenVINO:

* a plate **detector** (YOLOv9-tiny, 384x384) that returns plate boxes, and
* a plate **OCR** network (64x128) that reads up to ten characters from one
  cropped plate and guesses the issuing region.

Both come from the MIT-licensed ``open-image-models`` / ``fast-plate-ocr``
projects. The pre- and post-processing here mirrors theirs exactly; the
integration test checks the two agree on a real image.

What this cannot do: read a plate that is not legible. Below roughly 80-100
pixels of plate width the characters are simply not in the image, and no
network recovers them. Callers get a confidence and a width and decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from computer_vision.inference.openvino_runner import OpenVinoModel

DETECTOR_SIZE = 384
OCR_HEIGHT, OCR_WIDTH = 64, 128
#: Index = class id of the OCR head. ``_`` pads plates shorter than ten.
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD_CHAR = "_"
PLATE_SLOTS = 10

#: Region labels of the ``cct_s_v2_global`` OCR model, in output order.
REGIONS: Tuple[str, ...] = (
    "Albania", "Andorra", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bahrain", "Belarus", "Belgium", "Bosnia and Herzegovina", "Brazil", "Bulgaria",
    "Cambodia", "Canada", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece", "Guernsey",
    "Hungary", "Iceland", "Indonesia", "Ireland", "Israel", "Italy", "Latvia",
    "Liechtenstein", "Lithuania", "Luxembourg", "Malaysia", "Malta", "Mexico", "Moldova",
    "Monaco", "Montenegro", "Netherlands", "New Zealand", "North Macedonia", "Norway",
    "Poland", "Portugal", "Qatar", "Romania", "San Marino", "Serbia", "Singapore",
    "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Thailand", "Turkey",
    "United States", "Ukraine", "United Kingdom", "Vietnam", "Unknown",
)


@dataclass(frozen=True)
class PlateBox:
    """A plate located inside the image handed to :meth:`PlateReader.detect`."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1


@dataclass(frozen=True)
class PlateText:
    text: str
    #: Mean of the per-character probabilities, padding slots included.
    confidence: float
    region: str = ""


def letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize keeping aspect ratio, pad to ``size`` x ``size`` with grey."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2

    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return image, ratio, (pad_w, pad_h)


def decode_plate(probabilities: np.ndarray, region_scores: np.ndarray | None = None) -> PlateText:
    """Turn the OCR head's ``(slots, alphabet)`` probabilities into text."""
    slots = np.asarray(probabilities).reshape(PLATE_SLOTS, len(ALPHABET))
    indexes = slots.argmax(axis=-1)
    text = "".join(ALPHABET[i] for i in indexes).rstrip(PAD_CHAR)
    confidence = float(slots.max(axis=-1).mean())

    region = ""
    if region_scores is not None and np.size(region_scores) == len(REGIONS):
        region = REGIONS[int(np.argmax(region_scores))]
    return PlateText(text=text, confidence=confidence, region=region)


class PlateReader:
    """Plate detector + OCR behind two calls: :meth:`detect` and :meth:`read`."""

    def __init__(
        self,
        detector_model: str,
        ocr_model: str,
        *,
        detector_sha256: str | None = None,
        ocr_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._detector = OpenVinoModel(
            detector_model,
            sha256=detector_sha256,
            device=device,
            purpose="number-plate detector",
        )
        self._ocr = OpenVinoModel(
            ocr_model,
            sha256=ocr_sha256,
            device=device,
            input_shape=(1, OCR_HEIGHT, OCR_WIDTH, 3),
            purpose="number-plate OCR model",
        )

    def detect(self, image: np.ndarray, min_confidence: float = 0.4) -> List[PlateBox]:
        """Plates in a BGR image (normally one vehicle crop), best first."""
        if image is None or image.size == 0:
            return []
        boxed, ratio, (pad_w, pad_h) = letterbox(image, DETECTOR_SIZE)
        tensor = boxed.transpose((2, 0, 1))[::-1].astype(np.float32) / 255.0
        rows = self._detector(np.ascontiguousarray(tensor[None]))[0].reshape(-1, 7)

        height, width = image.shape[:2]
        plates: List[PlateBox] = []
        for _, x1, y1, x2, y2, _, score in rows:
            if score < min_confidence:
                continue
            box = PlateBox(
                x1=int(np.clip((x1 - pad_w) / ratio, 0, width)),
                y1=int(np.clip((y1 - pad_h) / ratio, 0, height)),
                x2=int(np.clip((x2 - pad_w) / ratio, 0, width)),
                y2=int(np.clip((y2 - pad_h) / ratio, 0, height)),
                confidence=float(score),
            )
            if box.x2 > box.x1 and box.y2 > box.y1:
                plates.append(box)
        plates.sort(key=lambda plate: plate.confidence, reverse=True)
        return plates

    def read(self, plate_image: np.ndarray) -> PlateText:
        """Text on one cropped BGR plate image."""
        if plate_image is None or plate_image.size == 0:
            return PlateText(text="", confidence=0.0)
        rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (OCR_WIDTH, OCR_HEIGHT), interpolation=cv2.INTER_LINEAR)
        outputs = self._ocr(np.ascontiguousarray(resized[None]).astype(np.uint8))
        region = outputs[1] if len(outputs) > 1 else None
        return decode_plate(outputs[0], region)
