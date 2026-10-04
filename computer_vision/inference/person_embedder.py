"""Person appearance embedding for re-identification.

Wraps the OpenCV Zoo *YoutuReID* network (Apache-2.0): a person crop goes in,
a 768-number description of how that person looks comes out. Two crops of the
same person give similar vectors; that similarity is all re-identification is.

It describes clothing, build and colour - not a face. That makes it work from
behind and at a distance, and also means two people in the same uniform look
alike to it, and one person who takes a coat off looks like someone new.
"""

from __future__ import annotations

import cv2
import numpy as np

from computer_vision.inference.openvino_runner import OpenVinoModel

INPUT_WIDTH, INPUT_HEIGHT = 128, 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / (float(np.linalg.norm(vector)) + 1e-9)


class PersonEmbedder:
    """Turns a BGR person crop into a unit-length appearance vector."""

    def __init__(self, model: str, *, sha256: str | None = None, device: str = "cpu") -> None:
        self._model = OpenVinoModel(
            model,
            sha256=sha256,
            device=device,
            input_shape=(1, 3, INPUT_HEIGHT, INPUT_WIDTH),
            purpose="person re-identification model",
        )

    def embed(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
        return normalise(self._model(np.ascontiguousarray(tensor))[0])
