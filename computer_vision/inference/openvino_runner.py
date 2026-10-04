"""A minimal OpenVINO model runner for the small auxiliary networks.

The YOLO detectors go through Ultralytics. The helper networks - plate
detector, plate OCR, person re-identification - are plain ONNX files with one
input and a couple of outputs, and need nothing more than "load, compile, run".
OpenVINO reads ONNX directly, so those modules add no runtime dependency beyond
the one the default detector already uses.

Unlike ``.pt`` weights, ONNX and OpenVINO IR are data, not pickles: loading
one does not execute code from the file. They are still pinned by SHA-256 when
the configuration gives one, because a swapped model is a wrong answer.
"""

from __future__ import annotations

import importlib.util
from typing import List, Sequence

import numpy as np

from computer_vision.inference.base import (
    BackendNotAvailableError,
    ModelLoadError,
    verify_model_file,
)

_DEVICE_ALIASES = {"cpu": "CPU", "intel:cpu": "CPU", "gpu": "GPU", "intel:gpu": "GPU", "auto": "AUTO"}


class OpenVinoModel:
    """One compiled network: call it with an array, get its outputs back."""

    def __init__(
        self,
        model_path: str,
        *,
        sha256: str | None = None,
        device: str = "cpu",
        input_shape: Sequence[int] | None = None,
        purpose: str = "model",
    ) -> None:
        if importlib.util.find_spec("openvino") is None:
            raise BackendNotAvailableError(
                f"The {purpose} needs the 'openvino' package.\n"
                "Install it with:  pip install openvino"
            )
        import openvino as ov

        path = verify_model_file(model_path, expected_sha256=sha256)
        core = ov.Core()
        try:
            model = core.read_model(path)
            if input_shape is not None:
                # Fix a dynamic batch dimension: a static shape compiles to a
                # faster network than a dynamic one.
                model.reshape(list(input_shape))
            target = _DEVICE_ALIASES.get(str(device).strip().lower(), str(device).upper())
            self._compiled = core.compile_model(
                model, target, {"PERFORMANCE_HINT": "LATENCY"}
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as ModelLoadError
            raise ModelLoadError(
                f"Could not load the {purpose} from {model_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._request = self._compiled.create_infer_request()
        self.path = path

    def __call__(self, array: np.ndarray) -> List[np.ndarray]:
        self._request.infer({0: array})
        return [
            self._request.get_output_tensor(index).data.copy()
            for index in range(len(self._compiled.outputs))
        ]
