"""SQLite storage for registered users and their face templates.

Like the engine's identity store, this keeps **embeddings only** - numeric
face templates, never photographs. A leaked database holds vectors that can be
compared against a face you already have, not a gallery of faces. Deleting a
user deletes every one of their templates in the same transaction.

One connection per call (SQLite connections are not shared across threads), WAL
mode so the pipeline reading and the dashboard writing do not block each other.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL COLLATE NOCASE UNIQUE,
    employee_id TEXT    NOT NULL DEFAULT '',
    role        TEXT    NOT NULL DEFAULT '',
    department  TEXT    NOT NULL DEFAULT '',
    notes       TEXT    NOT NULL DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS face_samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pose       TEXT    NOT NULL,
    embedding  BLOB    NOT NULL,
    dim        INTEGER NOT NULL,
    quality    REAL    NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_user ON face_samples(user_id);
"""

#: Editable text fields of a user, with the longest value each accepts.
USER_FIELDS: Dict[str, int] = {
    "name": 80,
    "employee_id": 40,
    "role": 60,
    "department": 60,
    "notes": 500,
}


class FaceDbError(Exception):
    """A rejected change; the message is safe to show the operator."""


class DuplicateUserError(FaceDbError):
    """Another user already has that name."""


class UserNotFoundError(FaceDbError):
    """No user with that id."""


@dataclass(frozen=True)
class UserRow:
    id: int
    name: str
    employee_id: str
    role: str
    department: str
    notes: str
    active: bool
    created_at: str
    updated_at: str
    #: ``{pose: sample count}`` for this user.
    poses: Dict[str, int]

    @property
    def sample_count(self) -> int:
        return sum(self.poses.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "employee_id": self.employee_id,
            "role": self.role,
            "department": self.department,
            "notes": self.notes,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "poses": dict(self.poses),
            "sample_count": self.sample_count,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_fields(raw: Dict[str, Any], *, require_name: bool) -> Dict[str, str]:
    """Validated text fields from a request body. Raises :class:`FaceDbError`."""
    out: Dict[str, str] = {}
    for key, limit in USER_FIELDS.items():
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise FaceDbError(f"'{key}' must be text")
        value = " ".join(value.split()) if key != "notes" else value.strip()
        if len(value) > limit:
            raise FaceDbError(f"'{key}' is longer than {limit} characters")
        out[key] = value
    if require_name and not out.get("name"):
        raise FaceDbError("A name is required")
    if "name" in out and not out["name"]:
        raise FaceDbError("The name must not be empty")
    return out


class FaceDatabase:
    """Users and face templates on disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        self._restrict()

    # ------------------------------------------------------------------ #

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _restrict(self) -> None:
        """Owner-only where the OS supports it: the file holds biometric templates."""
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - filesystem dependent
            pass

    def _poses(self, conn: sqlite3.Connection, user_id: int) -> Dict[str, int]:
        rows = conn.execute(
            "SELECT pose, COUNT(*) AS n FROM face_samples WHERE user_id = ? GROUP BY pose",
            (user_id,),
        ).fetchall()
        return {row["pose"]: int(row["n"]) for row in rows}

    def _row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> UserRow:
        return UserRow(
            id=int(row["id"]),
            name=row["name"],
            employee_id=row["employee_id"],
            role=row["role"],
            department=row["department"],
            notes=row["notes"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            poses=self._poses(conn, int(row["id"])),
        )

    # ------------------------------------------------------------------ #
    # Users
    # ------------------------------------------------------------------ #

    def list_users(self) -> List[UserRow]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY name COLLATE NOCASE").fetchall()
            return [self._row(conn, row) for row in rows]

    def get_user(self, user_id: int) -> UserRow:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserNotFoundError(f"No user with id {user_id}")
            return self._row(conn, row)

    def create_user(self, fields: Dict[str, str]) -> UserRow:
        now = _now()
        with self._write_lock, self._connect() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO users (name, employee_id, role, department, notes, active,"
                    " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        fields["name"],
                        fields.get("employee_id", ""),
                        fields.get("role", ""),
                        fields.get("department", ""),
                        fields.get("notes", ""),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateUserError(
                    f"A user named '{fields['name']}' already exists"
                ) from exc
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            return self._row(conn, row)

    def update_user(
        self, user_id: int, fields: Dict[str, str], active: Optional[bool] = None
    ) -> UserRow:
        sets: List[str] = []
        values: List[Any] = []
        for key in USER_FIELDS:
            if key in fields:
                sets.append(f"{key} = ?")
                values.append(fields[key])
        if active is not None:
            sets.append("active = ?")
            values.append(1 if active else 0)
        with self._write_lock, self._connect() as conn:
            if sets:
                sets.append("updated_at = ?")
                values.append(_now())
                try:
                    cursor = conn.execute(
                        f"UPDATE users SET {', '.join(sets)} WHERE id = ?",
                        (*values, user_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateUserError(
                        f"A user named '{fields.get('name', '')}' already exists"
                    ) from exc
                if cursor.rowcount == 0:
                    raise UserNotFoundError(f"No user with id {user_id}")
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserNotFoundError(f"No user with id {user_id}")
            return self._row(conn, row)

    def delete_user(self, user_id: int) -> None:
        """Delete a user and, by cascade, every face template of theirs."""
        with self._write_lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            if cursor.rowcount == 0:
                raise UserNotFoundError(f"No user with id {user_id}")

    # ------------------------------------------------------------------ #
    # Face templates
    # ------------------------------------------------------------------ #

    def add_sample(self, user_id: int, pose: str, embedding: np.ndarray, quality: float) -> None:
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        with self._write_lock, self._connect() as conn:
            if conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
                raise UserNotFoundError(f"No user with id {user_id}")
            conn.execute(
                "INSERT INTO face_samples (user_id, pose, embedding, dim, quality, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, pose, vector.tobytes(), int(vector.size), float(quality), _now()),
            )
            conn.execute("UPDATE users SET updated_at = ? WHERE id = ?", (_now(), user_id))

    def delete_samples(self, user_id: int, pose: Optional[str] = None) -> int:
        """Remove a user's templates (one pose, or all). Returns how many."""
        with self._write_lock, self._connect() as conn:
            if pose is None:
                cursor = conn.execute("DELETE FROM face_samples WHERE user_id = ?", (user_id,))
            else:
                cursor = conn.execute(
                    "DELETE FROM face_samples WHERE user_id = ? AND pose = ?", (user_id, pose)
                )
            conn.execute("UPDATE users SET updated_at = ? WHERE id = ?", (_now(), user_id))
            return int(cursor.rowcount)

    def samples_of(self, user_id: int) -> List[Tuple[str, np.ndarray]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pose, embedding FROM face_samples WHERE user_id = ?", (user_id,)
            ).fetchall()
        return [(r["pose"], np.frombuffer(r["embedding"], dtype=np.float32).copy()) for r in rows]

    def active_identities(self) -> List[Tuple[str, List[np.ndarray]]]:
        """``(name, embeddings)`` for every active user that has templates."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT u.name AS name, s.embedding AS embedding FROM face_samples s"
                " JOIN users u ON u.id = s.user_id WHERE u.active = 1 ORDER BY u.id"
            ).fetchall()
        grouped: Dict[str, List[np.ndarray]] = {}
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            grouped.setdefault(row["name"], []).append(vector)
        return list(grouped.items())

    def embedding_dim(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute("SELECT dim FROM face_samples LIMIT 1").fetchone()
        return int(row["dim"]) if row else None
