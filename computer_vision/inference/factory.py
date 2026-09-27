"""Detector backend factory.

Mirrors ``computer_vision.sources.factory``: the only place that maps a config value to a
concrete backend class, and it imports a backend lazily so an unused runtime is
never loaded into the process.

It is also where model *sharing* is decided. Modules whose model configuration
matches receive the same :class:`~computer_vision.inference.shared.SharedDetector`, so the
weights load once and each frame is inferred once no matter how many detectors
are enabled. See ``computer_vision/inference/shared.py`` for why that is safe.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List, Tuple

from computer_vision.inference.base import BackendError, DetectorBackend
from computer_vision.inference.shared import SharedDetector, share_key
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

#: backend name -> "import.path:ClassName"
BACKENDS: Dict[str, str] = {
    "ultralytics": "computer_vision.inference.ultralytics_backend:UltralyticsDetector",
    # Planned, so a closed-source deployment can avoid the AGPL runtime:
    # "onnx":       "computer_vision.inference.onnx_backend:OnnxDetector",
    # "opencv_dnn": "computer_vision.inference.opencv_dnn_backend:OpenCVDnnDetector",
}

_shared_cache: Dict[Tuple[Any, ...], SharedDetector] = {}
_cache_lock = threading.Lock()


def available_backends() -> List[str]:
    return sorted(BACKENDS)


def reset_detector_cache() -> None:
    """Drop every shared detector. For tests and for a clean reconfigure."""
    with _cache_lock:
        _shared_cache.clear()


def shared_detector_count() -> int:
    """How many distinct models are currently shared. Used by tests."""
    with _cache_lock:
        return len(_shared_cache)


def _resolve_class(name: str) -> type[DetectorBackend]:
    import_path = BACKENDS.get(name)
    if import_path is None:
        raise BackendError(
            f"Unknown detector backend {name!r}. Available: {available_backends()}"
        )

    module_path, _, class_name = import_path.partition(":")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise BackendError(
            f"Could not load detector backend {name!r} from {import_path!r}: {exc}"
        ) from exc


def create_detector(name: str, params: Dict[str, Any] | None = None) -> DetectorBackend:
    """Build (but do not load) a detector backend.

    When ``share_model`` is true (the default), detectors with matching model
    configuration are shared: the weights load once and each frame is inferred
    once, however many modules are enabled.

    Raises:
        BackendError: the name is unknown or the adapter cannot be imported.
    """
    params = dict(params or {})
    backend_cls = _resolve_class(name)

    if not params.pop("share_model", True):
        logger.debug("Created unshared detector backend '%s'", name)
        return backend_cls(params)

    key = share_key(name, params)
    with _cache_lock:
        existing = _shared_cache.get(key)
        if existing is not None:
            logger.debug("Sharing existing detector for %s", params.get("model"))
            return existing

        shared = SharedDetector(backend_cls(params))
        _shared_cache[key] = shared
        return shared
