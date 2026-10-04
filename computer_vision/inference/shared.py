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

With several cameras in one process the same detector is shared by all of
them. Two things keep that correct: inference is serialised (the underlying
runtimes are not safe to call from several threads at once, and one CPU
inference at a time is no slower than two interleaved), and the per-frame memo
is kept per thread - each camera runs on its own thread, and a shared memo
would be overwritten by the other camera between two modules of the same frame.

Each module still filters the shared results down to the labels it owns, so
they keep completely separate outputs. Sharing is an implementation detail
below the module boundary - no module knows it is happening.

Measured: restricting a YOLO model to specific classes saves nothing (+0.2%,
within noise) because the filter is applied after NMS. So the shared inference
covers every class and modules filter by label afterwards, at no cost.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

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


#: What a backend uses when a param is left out. Two modules must share when
#: one spells a default out (``imgsz: 640``) and the other omits it - comparing
#: the raw values treated those as different models, so the same weights were
#: loaded twice and every frame was inferred twice.
_IDENTITY_DEFAULTS: Dict[str, Any] = {
    "device": "cpu",
    "imgsz": 640,
    "model_confidence": 0.25,
    "iou": 0.45,
    "half": False,
    "classes": None,
}


def _normalise(name: str, value: Any) -> Any:
    """One canonical form per identity param, so ``640`` and ``"640"`` agree."""
    if value is None:
        value = _IDENTITY_DEFAULTS.get(name)
    if value is None:
        return None
    try:
        if name == "device":
            return str(value).strip().lower()
        if name == "imgsz":
            if isinstance(value, (list, tuple)):
                return tuple(int(side) for side in value)
            return int(value)
        if name in ("model_confidence", "iou"):
            return round(float(value), 6)
        if name == "half":
            return bool(value)
        if name == "classes":
            return tuple(int(c) for c in value) or None
    except (TypeError, ValueError):
        pass  # leave a malformed value as-is; the backend reports it on load
    return tuple(value) if isinstance(value, list) else value


def share_key(backend_name: str, params: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of a model configuration, for deciding what can be shared."""
    values: List[Any] = [backend_name]
    for name in _IDENTITY_PARAMS:
        values.append(_normalise(name, params.get(name)))
    return tuple(values)


class SharedDetector(DetectorBackend):
    """Reference-counted, frame-memoising wrapper around a detector backend."""

    name = "shared"

    def __init__(self, inner: DetectorBackend) -> None:
        super().__init__(inner.params)
        self._inner = inner
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._users = 0

        # Memo for the frame currently in flight, one per calling thread. A
        # strong reference to the frame is held deliberately: it makes the `is`
        # identity check below sound, because the object cannot be collected
        # (and its address reused by a different array) while it is cached.
        self._memo = threading.local()

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
        memo = self._memo
        if getattr(memo, "frame", None) is frame:
            with self._lock:
                self.cache_hits += 1
            return list(memo.results)

        with self._infer_lock:
            results = self._inner.infer(frame)

        memo.frame = frame
        memo.results = list(results)
        with self._lock:
            self.inference_count += 1

        return list(results)

    def _forget_frame(self) -> None:
        # Other threads' memos cannot be reached from here; replacing the
        # holder makes every thread start from an empty memo.
        self._memo = threading.local()

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
