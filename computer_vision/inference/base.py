"""Detector backend contract.

An AI module must not depend on an inference runtime directly. It depends on
this small interface, and a backend adapter implements it. Swapping Ultralytics
for ONNX Runtime, OpenVINO or TensorRT then means adding one file here and
changing one config value - no module, pipeline or source code changes.

``RawDetection`` is deliberately primitive (no ``BBox``, no ``Detection``):
backends speak model language, modules speak VisionGuard language, and the
module does the translation.
"""

from __future__ import annotations

import hashlib
import hmac
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Sequence, Tuple

import numpy as np


class BackendError(Exception):
    """Base class for inference-backend failures."""


class BackendNotAvailableError(BackendError):
    """The backend's runtime package is not installed."""


class ModelLoadError(BackendError):
    """The model file is missing, corrupt or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(path: str | Path) -> str:
    """SHA-256 identifying a model: a file's own hash, or a directory's.

    An exported model (OpenVINO IR, for example) is a directory of several
    files. Its digest covers every file's relative name and content, in name
    order, so adding, removing, renaming or editing any of them changes it.
    """
    target = Path(path)
    if target.is_file():
        return _sha256(target)

    digest = hashlib.sha256()
    for item in sorted(p for p in target.rglob("*") if p.is_file()):
        digest.update(item.relative_to(target).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_model_file(
    model_ref: str,
    expected_sha256: str | None = None,
    allow_download: bool = False,
    download: Callable[[str], str] | None = None,
) -> str:
    """Check a weights file *before* a pickle-based loader opens it.

    ``.pt`` weights are Python pickles: loading one executes code from it.
    So a weights file must be the file you meant, from where you meant:

    * A missing file is an error instead of an implicit internet download,
      unless ``allow_download`` is set. Even then the file is fetched with
      ``download`` (which must only download, never load) so it can be
      verified before anything deserialises it.
    * When ``expected_sha256`` is given, the file must match it exactly.

    Returns:
        The path to hand to the loader (the downloaded location if fetched).

    Raises:
        ModelLoadError: missing file (download not allowed / failed) or a
            hash mismatch.
    """
    path = Path(model_ref)
    looks_like_file = bool(path.suffix) or len(path.parts) > 1

    if path.is_dir():
        # An exported model directory. Never downloaded; only checked.
        if not any(path.iterdir()):
            raise ModelLoadError(f"Model directory is empty: {path}")
        if expected_sha256:
            expected = str(expected_sha256).strip().lower()
            actual = model_digest(path)
            if not hmac.compare_digest(actual, expected):
                raise ModelLoadError(
                    f"Model directory {path} failed its integrity check: digest "
                    f"is {actual}, expected {expected}. A file in it was changed, "
                    "added or removed; refusing to load it."
                )
        return str(path)

    if not path.is_file():
        if not looks_like_file:
            # A runtime model *name*, not a path; nothing local to verify.
            return model_ref
        if not allow_download:
            raise ModelLoadError(
                f"Model file not found: {path}\n"
                f"(resolved from working directory {Path.cwd()})\n"
                "VisionGuard does not download weights implicitly, because a "
                "weights file is executable code. Either place the file at "
                "that path (see computer_vision/Model/README.md), or set "
                "'allow_download: true' on the module to fetch it once "
                "(its model_sha256 is still checked before loading)."
            )
        if download is None:
            raise ModelLoadError(f"Model file not found and no downloader available: {path}")
        try:
            path = Path(download(model_ref))
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise ModelLoadError(f"Could not download model {model_ref!r}: {exc}") from exc
        if not path.is_file():
            raise ModelLoadError(f"Download of {model_ref!r} produced no file")

    if expected_sha256:
        expected = str(expected_sha256).strip().lower()
        actual = _sha256(path)
        if not hmac.compare_digest(actual, expected):
            raise ModelLoadError(
                f"Model file {path} failed its integrity check: sha256 is "
                f"{actual}, expected {expected}. The file may be corrupt or "
                "replaced; refusing to load it."
            )
    return str(path)


@dataclass(frozen=True)
class RawDetection:
    """One detection straight out of a model, in frame pixel coordinates."""

    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    class_id: int = -1


class DetectorBackend(ABC):
    """An object-detection runtime adapter."""

    #: Value used in ``config.yaml`` to select this backend.
    name: ClassVar[str] = ""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.

        Called only by an enabled module's ``initialize()``, which is how
        "disabled modules must not load their models" is upheld.

        Raises:
            BackendNotAvailableError: the runtime package is missing.
            ModelLoadError: the model could not be loaded.
        """

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[RawDetection]:
        """Run inference on one BGR frame."""

    def infer_batch(self, frames: Sequence[np.ndarray]) -> List[List[RawDetection]]:
        """Run inference on several frames; one result list per frame, in order.

        The default runs them one by one. A runtime that can batch (one
        forward pass for all frames) overrides it - that is what lets several
        cameras share one GPU model without queueing behind each other.
        """
        return [self.infer(frame) for frame in frames]

    def unload(self) -> None:
        """Release the model. Safe to call more than once."""
        self._loaded = False

    @property
    def description(self) -> str:
        return f"{type(self).__name__}"

    def _resolve_labels(self, class_ids: Sequence[int], names: Dict[int, str]) -> List[str]:
        return [names.get(int(cid), str(cid)) for cid in class_ids]

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} {state}>"
