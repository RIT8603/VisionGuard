"""Identity storage for face recognition.

Isolated from the AI modules on purpose. Recognition asks "which enrolled
identity does this embedding match?" and the store answers; neither knows how
the other works, and the store can be replaced (a database, a directory
service) without touching a module.

See :mod:`app.identity.store` for the privacy properties this deliberately
has - chiefly that it holds embeddings and never images.
"""

from app.identity.store import (
    Identity,
    IdentityMatch,
    IdentityStore,
    IdentityStoreError,
)

__all__ = ["Identity", "IdentityMatch", "IdentityStore", "IdentityStoreError"]
