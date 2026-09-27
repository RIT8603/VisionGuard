"""Enrolled identities, stored as embeddings.

Design decisions, and why - this is the part of VisionGuard where getting the
data model wrong has consequences beyond a refactor (Agent.md §24).

**Embeddings only, never images.** The store holds float vectors. It never
writes a face crop, a photograph, or anything renderable. A leaked store
exposes numbers that can be compared against a face you already have; it does
not hand anyone a gallery of faces. Enrolment images are the caller's
responsibility and are never persisted here.

**One file, explicit and portable.** A single ``.npz`` holding names,
embeddings and metadata. It can be inspected, diffed, backed up and *deleted*
in one operation - deletion being the operation a subject is most likely to
ask for.

**Several embeddings per identity.** Faces vary with pose and light; matching
takes the best similarity across an identity's samples rather than an average,
which would blur distinct poses into a vector resembling neither.

**No implicit enrolment.** Nothing is ever added by seeing it. An identity
exists because someone enrolled it deliberately.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from computer_vision.inference.face_base import cosine_similarity
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

STORE_VERSION = 1


class IdentityStoreError(Exception):
    """The identity store is missing, malformed or incompatible."""


@dataclass
class Identity:
    """One enrolled person and their embedding samples."""

    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    enrolled_at: str = ""

    @property
    def sample_count(self) -> int:
        return len(self.embeddings)


@dataclass(frozen=True)
class IdentityMatch:
    """The best candidate for an embedding."""

    name: str
    similarity: float
    matched: bool

    @property
    def label(self) -> str:
        return self.name if self.matched else "unknown"


class IdentityStore:
    """An in-memory set of enrolled identities, persisted as one ``.npz``.

    Thread-safe for concurrent reads; writes take a lock.
    """

    def __init__(self, threshold: float = 0.363) -> None:
        self._identities: Dict[str, Identity] = {}
        self._lock = threading.RLock()
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ #
    # Contents
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        with self._lock:
            return len(self._identities)

    @property
    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._identities)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return sum(i.sample_count for i in self._identities.values())

    def get(self, name: str) -> Optional[Identity]:
        with self._lock:
            return self._identities.get(name)

    def is_empty(self) -> bool:
        return len(self) == 0

    # ------------------------------------------------------------------ #
    # Enrolment
    # ------------------------------------------------------------------ #

    def enroll(self, name: str, embedding: np.ndarray) -> Identity:
        """Add one embedding sample for ``name``, creating it if needed."""
        clean = str(name).strip()
        if not clean:
            raise IdentityStoreError("Identity name must not be empty")

        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size == 0:
            raise IdentityStoreError(f"Embedding for '{clean}' is empty")
        if not np.all(np.isfinite(vector)):
            raise IdentityStoreError(f"Embedding for '{clean}' contains NaN or inf")

        with self._lock:
            self._check_dimension(vector.size)
            identity = self._identities.get(clean)
            if identity is None:
                identity = Identity(
                    name=clean, enrolled_at=datetime.now(timezone.utc).isoformat()
                )
                self._identities[clean] = identity
            identity.embeddings.append(vector)
            return identity

    def remove(self, name: str) -> bool:
        """Delete an identity and every sample of it. Returns True if present.

        The operation a subject is most likely to request, so it is a first
        class method rather than something a caller has to assemble.
        """
        with self._lock:
            existed = self._identities.pop(str(name).strip(), None) is not None
        if existed:
            logger.info("Removed identity '%s' from the store", name)
        return existed

    def clear(self) -> None:
        with self._lock:
            self._identities.clear()

    def _check_dimension(self, size: int) -> None:
        for identity in self._identities.values():
            for existing in identity.embeddings:
                if existing.size != size:
                    raise IdentityStoreError(
                        f"Embedding dimension {size} does not match the stored "
                        f"{existing.size}. The store was built with a different "
                        "face model; re-enrol against the current one."
                    )
                return

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #

    def match(self, embedding: np.ndarray, threshold: Optional[float] = None) -> IdentityMatch:
        """Best identity for ``embedding``.

        Returns the nearest candidate along with whether it cleared the
        threshold - the caller gets the score either way, so "unknown, but it
        was close" is distinguishable from "unknown, nothing like it".
        """
        cutoff = self.threshold if threshold is None else float(threshold)
        vector = np.asarray(embedding, dtype=np.float32).ravel()

        best_name, best_score = "", -1.0
        with self._lock:
            for identity in self._identities.values():
                for sample in identity.embeddings:
                    if sample.size != vector.size:
                        continue
                    score = cosine_similarity(vector, sample)
                    if score > best_score:
                        best_name, best_score = identity.name, score

        if not best_name:
            return IdentityMatch(name="", similarity=0.0, matched=False)
        return IdentityMatch(
            name=best_name, similarity=best_score, matched=best_score >= cutoff
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Write the store to one ``.npz``. Never writes image data."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        names: List[str] = []
        enrolled: List[str] = []
        vectors: List[np.ndarray] = []
        with self._lock:
            for identity in self._identities.values():
                for sample in identity.embeddings:
                    names.append(identity.name)
                    enrolled.append(identity.enrolled_at)
                    vectors.append(sample)

        stacked = (
            np.stack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, 0), dtype=np.float32)
        )
        # Unicode arrays, not dtype=object: object arrays would force
        # allow_pickle=True on load, which turns a store file into arbitrary
        # code execution. Nothing here needs pickle.
        np.savez_compressed(
            target,
            version=np.array([STORE_VERSION], dtype=np.int32),
            names=np.array(names, dtype=np.str_),
            enrolled_at=np.array(enrolled, dtype=np.str_),
            embeddings=stacked,
        )
        logger.info(
            "Saved %d identities (%d samples) to %s",
            len(self),
            len(vectors),
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.363) -> "IdentityStore":
        """Read a store written by :meth:`save`."""
        source = Path(path)
        if not source.is_file():
            raise IdentityStoreError(
                f"Identity store not found: {source}\n"
                "Enrol identities first, or point modules.face_recognition."
                "identity_store at an existing file."
            )

        try:
            # allow_pickle stays False: a store path comes from configuration,
            # and a pickled array in a swapped-in file would execute on load.
            with np.load(source, allow_pickle=False) as data:
                version = int(data["version"][0]) if "version" in data else 0
                if version != STORE_VERSION:
                    raise IdentityStoreError(
                        f"Identity store {source} is version {version}, "
                        f"this build expects {STORE_VERSION}."
                    )
                names = [str(n) for n in data["names"]]
                enrolled = [str(t) for t in data["enrolled_at"]]
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        except IdentityStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise IdentityStoreError(f"Could not read {source}: {exc}") from exc

        store = cls(threshold=threshold)
        for index, name in enumerate(names):
            identity = store._identities.setdefault(
                name, Identity(name=name, enrolled_at=enrolled[index])
            )
            identity.embeddings.append(embeddings[index])

        logger.info(
            "Loaded %d identities (%d samples) from %s",
            len(store),
            len(names),
            source,
        )
        return store

    def describe(self) -> Dict[str, object]:
        """Summary safe to log: counts, not vectors."""
        with self._lock:
            return {
                "identities": len(self._identities),
                "samples": sum(i.sample_count for i in self._identities.values()),
                "threshold": self.threshold,
            }

    def __repr__(self) -> str:
        return f"<IdentityStore {len(self)} identities, {self.sample_count} samples>"
