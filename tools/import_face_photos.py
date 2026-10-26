"""Register users from a face-photo export into the face registry (one-off).

Another VisionGuard install can export its registry as photos::

    face-photos/
      manifest.csv          file, sample_id, user_id, name, employee_id, pose, quality, ...
      photos/<id>_<name>/<sample>_<pose>.jpg

This tool enrols those people here through the *same* code the dashboard uses
(``FaceService.add_sample``), so every photo passes the usual quality, pose and
"is this already somebody else?" checks, and only embeddings are stored.

Usage, from the project root::

    python -m tools.import_face_photos "data/faces/New user face data/face-photos" --dry-run
    python -m tools.import_face_photos "data/faces/New user face data/face-photos" \\
        --source-db "data/faces/New user face data/faces (1).db" --import-embeddings

* Existing users are never changed: a person already registered here (same
  employee id, or same name when there is no employee id) is skipped.
* Photos are tried front views first, best quality first, and filed under the view the face is
  actually in (``auto``), up to the registry's per-view limit.
* ``--source-db`` (the exporting install's ``faces.db``) supplies role,
  department and notes. With ``--import-embeddings`` it also brings over users
  who have stored templates but no photos in the export; those templates are
  checked against everyone already registered before being added.
* A user who ends up with no accepted face is not left behind as an empty record.

Stop the backend first, or restart it afterwards: a running server keeps its
own copy of the live face store and only re-reads the database on restart.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.persistence.face_db import FaceDatabase, FaceDbError
from backend.services.faces import MAX_SAMPLES_PER_POSE, POSES, FaceService, FaceServiceError
from computer_vision.core.config import load_config


def _source_users(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        users = {int(r["id"]): dict(r) for r in conn.execute("SELECT * FROM users")}
        for uid, user in users.items():
            user["samples"] = [
                (r["pose"], np.frombuffer(r["embedding"], dtype=np.float32).copy(), float(r["quality"] or 0.0))
                for r in conn.execute(
                    "SELECT pose, embedding, quality FROM face_samples WHERE user_id = ? ORDER BY quality DESC",
                    (uid,),
                )
            ]
    finally:
        conn.close()
    return users


def _find_existing(service: FaceService, name: str, employee_id: str):
    for user in service.db.list_users():
        if employee_id and user.employee_id and user.employee_id.lower() == employee_id.lower():
            return user
        if not employee_id and user.name.lower() == name.lower():
            return user
    return None


def _fields(name: str, employee_id: str, source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {"name": name, "employee_id": employee_id}
    if source:
        for key in ("role", "department", "notes"):
            if source.get(key):
                fields[key] = source[key]
    return fields


def _enrol_photos(service: FaceService, user_id: int, photos: List[Path]) -> Dict[str, Any]:
    accepted: Dict[str, int] = defaultdict(int)
    rejected: Dict[str, int] = defaultdict(int)
    for photo in photos:
        try:
            result = service.add_sample(user_id, "auto", photo.read_bytes())
            accepted[result["pose"]] += 1
        except FaceServiceError as exc:
            rejected[str(exc).split(" - ")[0].split(". ")[0]] += 1
    return {"accepted": dict(accepted), "rejected": dict(rejected)}


def _import_embeddings(service: FaceService, user_id: int, samples) -> Dict[str, Any]:
    user = service.db.get_user(user_id)
    per_pose: Dict[str, int] = defaultdict(int)
    rejected: Dict[str, int] = defaultdict(int)
    for pose, vector, quality in samples:
        if pose not in POSES or per_pose[pose] >= MAX_SAMPLES_PER_POSE:
            continue
        try:
            service._check_identity(user, vector)
        except FaceServiceError as exc:
            rejected[str(exc)] += 1
            continue
        service.db.add_sample(user_id, pose, vector, quality)
        per_pose[pose] += 1
    service.rebuild_live_store()
    return {"accepted": dict(per_pose), "rejected": dict(rejected)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export", help="the face-photos directory (contains manifest.csv and photos/)")
    parser.add_argument("--source-db", help="the exporting install's faces.db (extra fields, embeddings)")
    parser.add_argument("--import-embeddings", action="store_true",
                        help="also register --source-db users that have templates but no photos")
    parser.add_argument("--config", help="VisionGuard config (default configs/config.yaml)")
    parser.add_argument("--db", help="target faces.db (default: dashboard.face_db from the config)")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    args = parser.parse_args(argv)

    export = Path(args.export)
    manifest = export / "manifest.csv"
    if not manifest.is_file():
        print(f"No manifest.csv in {export}", file=sys.stderr)
        return 1

    people: Dict[int, Dict[str, Any]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            person = people.setdefault(int(row["user_id"]), {
                "name": row["name"].strip(), "employee_id": row["employee_id"].strip(), "photos": [],
            })
            path = export / row["file"]
            if path.is_file():
                person["photos"].append((row.get("pose", ""), float(row.get("quality") or 0.0), path))

    source = _source_users(args.source_db)
    config = load_config(args.config)
    db_path = args.db or config.dashboard.face_db
    service = FaceService.from_config(config)
    service.db = FaceDatabase(db_path)
    service.rebuild_live_store()
    print(f"Target registry: {db_path} ({len(service.db.list_users())} users before import)")

    plan = []
    for uid, person in sorted(people.items()):
        # Front views first: they are the most distinctive, so the later side
        # views are checked against the user's own front templates.
        photos = [p for _, _, p in sorted(person["photos"], key=lambda x: (x[0] != "front", -x[1]))]
        if not photos and args.import_embeddings and source.get(uid, {}).get("samples"):
            continue  # registered from its stored templates below
        plan.append((uid, person["name"], person["employee_id"], photos, None))
    if args.import_embeddings:
        for uid, user in sorted(source.items()):
            if uid in people and people[uid]["photos"]:
                continue
            if user["samples"]:
                plan.append((uid, user["name"], user["employee_id"] or "", [], user["samples"]))

    summary = []
    for uid, name, employee_id, photos, samples in plan:
        existing = _find_existing(service, name, employee_id)
        if existing is not None:
            summary.append((name, employee_id, "skipped: already registered", {}))
            continue
        if not photos and not samples:
            summary.append((name, employee_id, "skipped: no photos in the export", {}))
            continue
        what = f"{len(photos)} photos" if photos else f"{len(samples)} stored templates"
        if args.dry_run:
            summary.append((name, employee_id, f"would register from {what}", {}))
            continue
        try:
            user = service.create_user(_fields(name, employee_id, source.get(uid)))
        except FaceDbError as exc:
            summary.append((name, employee_id, f"skipped: {exc}", {}))
            continue
        result = _enrol_photos(service, user["id"], photos) if photos else _import_embeddings(service, user["id"], samples)
        if not result["accepted"]:
            service.delete_user(user["id"])
            summary.append((name, employee_id, "not registered: no face accepted", result))
            continue
        final = service.get_user(user["id"])
        state = "registered" if final["enrolled"] else f"registered, missing views {final['missing_poses']}"
        summary.append((name, employee_id, f"{state} (id {user['id']}, from {what})", result))

    for name, employee_id, outcome, result in summary:
        print(f"- {name} [{employee_id}]: {outcome}")
        if result.get("accepted"):
            print(f"    accepted per view: {result['accepted']}")
        if result.get("rejected"):
            print(f"    rejected: {result['rejected']}")
    print(f"Registry now has {len(service.db.list_users())} users.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
