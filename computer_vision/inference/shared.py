"""Model sharing: one load, one inference, many modules.

Agent.md §9: *"Do NOT duplicate model inference unnecessarily."*

Two detection modules configured with the same weights - say
``person_detection`` and ``vehicle_detection`` both on ``yolo11n.pt`` - would
otherwise load that model twice and run the same backbone over the same pixels
twice per frame. The second pass produces identical numbers at full cost.

:class:`SharedDetector` fixes both halves:

* **One load.** The factory hands the same instance to every module whose
  model configuration matches, reference-counted so the first ``cleanup()``
  does not pull the model out from under the others.
* **One inference.** Results are memoised for the frame currently being
  processed, so the second module through gets the first module's output.

Each module still filters the shared results down to the labels it owns, so
they keep completely separate outputs. Sharing is an implementation detail
below the module boundary - no module knows it is happening.

Measured: restricting a YOLO model to specific classes saves nothing (+0.2%,
within noise) because the filter is applied after NMS. So the shared inference
covers every class and modules filter by label afterwards, at no cost.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from computer_vision.inference.base import DetectorBackend, RawDetection
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

#: Params that change what a model produces. Anything else (label lists,
#: confidence thresholds, box-size limits) is applied by the module afterwards
#: and therefore does not prevent sharing.
_IDENTITY_PARAMS = (
    "model",
    "device",
    "imgsz",
    "model_confidence",
    "iou",
    "half",
    "classes",
)


def share_key(backend_name: str, params: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of a model configuration, for deciding what can be shared."""
    values: List[Any] = [backend_name]
    for name in _IDENTITY_PARAMS:
        value = params.get(name)
        if isinstance(value, list):
            value = tuple(value)
        values.append(value)
    return tuple(values)


class SharedDetector(DetectorBackend):
    """Reference-counted, frame-memoising wrapper around a detector backend."""

    name = "shared"

    def __init__(self, inner: DetectorBackend) -> None:
        super().__init__(inner.params)
        self._inner = inner
        self._lock = threading.Lock()
        self._users = 0

        # Memo for the frame currently in flight. A strong reference to the
        # frame is held deliberately: it makes the `is` identity check below
        # sound, because the object cannot be collected (and its address
        # reused by a different array) while it is cached.
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_results: List[RawDetection] = []

        self.inference_count = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def users(self) -> int:
        """How many modules currently hold this detector."""
        return self._users

    def load(self) -> None:
        with self._lock:
            self._users += 1
            if self._inner.is_loaded:
                logger.info(
                    "Reusing already-loaded model %s (%d modules sharing it)",
                    self._inner.description,
                    self._users,
                )
                self._loaded = True
                return

        self._inner.load()
        self._loaded = True

    def unload(self) -> None:
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users > 0:
                # Another module is still using it; leave the model resident.
                return

        self._inner.unload()
        self._loaded = False
        self._forget_frame()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def infer(self, frame: np.ndarray) -> List[RawDetection]:
        """Run the model, or return this frame's already-computed results."""
        with self._lock:
            if self._cached_frame is frame:
                self.cache_hits += 1
                return list(self._cached_results)

        results = self._inner.infer(frame)

        with self._lock:
            self._cached_frame = frame
            self._cached_results = list(results)
            self.inference_count += 1

        return list(results)

    def _forget_frame(self) -> None:
        self._cached_frame = None
        self._cached_results = []

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def inner(self) -> DetectorBackend:
        return self._inner

    @property
    def description(self) -> str:
        return f"{self._inner.description} (shared x{self._users})"

    def stats(self) -> Dict[str, int]:
        return {
            "users": self._users,
            "inferences": self.inference_count,
            "cache_hits": self.cache_hits,
        }

    def __repr__(self) -> str:
        return f"<SharedDetector {self._inner.description} users={self._users}>"
