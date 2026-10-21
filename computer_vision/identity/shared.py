"""The process-wide, live identity store.

Enrolment happens in the dashboard while the pipeline is running, so the
recognition module cannot read a file once at start-up and be done: a user
registered a second ago has to be recognisable on the very next frame, without
rebuilding the modules.

The store is therefore a single shared object. The backend's face service
*writes* it (replacing its contents whenever a user is added, edited or
removed); the recognition module *reads* it. ``computer_vision`` stays free of
any web or database code - it only knows this tiny registry, and the
:class:`IdentityStore` it hands back already locks around its own mutations.
"""

from __future__ import annotations

import threading
from typing import Iterable, List, Tuple

import numpy as np

from computer_vision.identity.store import IdentityStore

_lock = threading.Lock()
_store = IdentityStore()
#: Bumped on every replacement, so a camera running in another process can tell
#: that it needs the new contents.
_version = 0


def get_shared_store() -> IdentityStore:
    """The live store. The same object for the life of the process."""
    return _store


def replace_shared_store(
    identities: Iterable[Tuple[str, Iterable[np.ndarray]]], threshold: float | None = None
) -> IdentityStore:
    """Swap the live store's contents for ``(name, embeddings)`` pairs.

    The new contents are built off to the side and applied in one step, so a
    frame being recognised never sees a half-rebuilt store.
    """
    fresh = IdentityStore()
    for name, embeddings in identities:
        for embedding in embeddings:
            fresh.enroll(name, embedding)

    global _version
    with _lock:
        with _store._lock:  # noqa: SLF001 - same package, one atomic swap
            _store._identities = fresh._identities
            if threshold is not None:
                _store.threshold = float(threshold)
        _version += 1
    return _store


def store_version() -> int:
    """Changes whenever the live store's contents are replaced."""
    return _version


def snapshot_identities() -> Tuple[List[Tuple[str, List[np.ndarray]]], float]:
    """``((name, embeddings) pairs, threshold)``: everything needed to rebuild the
    live store elsewhere - in a camera's worker process."""
    with _store._lock:  # noqa: SLF001
        pairs = [
            (identity.name, [vec.copy() for vec in identity.embeddings])
            for identity in _store._identities.values()  # noqa: SLF001
        ]
        return pairs, _store.threshold
