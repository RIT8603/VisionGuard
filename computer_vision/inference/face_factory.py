"""Face backend factory.

Mirrors ``computer_vision.inference.factory``: the only place mapping a config value to a
concrete face backend, importing it lazily so an unused runtime never loads.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from computer_vision.inference.face_base import BackendError, FaceDetectorBackend, FaceEmbedderBackend

FACE_DETECTORS: Dict[str, str] = {
    "yunet": "computer_vision.inference.opencv_face:YuNetFaceDetector",
    "yoloface": "computer_vision.inference.yolo_face:YoloFaceDetector",
}

FACE_EMBEDDERS: Dict[str, str] = {
    "sface": "computer_vision.inference.opencv_face:SFaceEmbedder",
}


def available_face_detectors() -> List[str]:
    return sorted(FACE_DETECTORS)


def available_face_embedders() -> List[str]:
    return sorted(FACE_EMBEDDERS)


def _build(catalogue: Dict[str, str], name: str, kind: str, params: Dict[str, Any]):
    import_path = catalogue.get(name)
    if import_path is None:
        raise BackendError(
            f"Unknown {kind} backend {name!r}. Available: {sorted(catalogue)}"
        )

    module_path, _, class_name = import_path.partition(":")
    try:
        module = importlib.import_module(module_path)
        backend_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise BackendError(
            f"Could not load {kind} backend {name!r} from {import_path!r}: {exc}"
        ) from exc

    return backend_cls(params)


def create_face_detector(
    name: str, params: Dict[str, Any] | None = None
) -> FaceDetectorBackend:
    """Build (but do not load) a face detector."""
    return _build(FACE_DETECTORS, name, "face detector", dict(params or {}))


def create_face_embedder(
    name: str, params: Dict[str, Any] | None = None
) -> FaceEmbedderBackend:
    """Build (but do not load) a face embedder."""
    return _build(FACE_EMBEDDERS, name, "face embedder", dict(params or {}))
