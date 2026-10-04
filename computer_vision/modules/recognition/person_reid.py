"""Person Re-Identification module.

One responsibility: recognise that a person being tracked now is the same
person seen earlier - after they left and came back, or on another camera.

A tracker's ``#7`` means "the same box as last frame". The moment a person
walks behind a pillar, or out of this camera and into the next, that id is
gone. This module gives each person a **global id** from how they look, and
reuses it when the same appearance turns up again.

How it works, per tracked person:

1. take a few crops as the track goes by and average their appearance vectors
   (one crop can be a bad angle; three are steadier),
2. compare that against every identity in the gallery, and
3. above the similarity threshold, adopt that identity and raise
   ``person_reidentified``; otherwise enrol a new one.

**Multi-camera.** Point every camera at the same ``gallery_path`` and they
share one gallery, so a person enrolled by the gate camera is recognised by the
corridor camera. That holds whether the cameras are separate VisionGuard
processes or the several cameras of one multi-camera run.

It matches on clothing and build, not on a face. People dressed alike are
confused; a person who changes clothes is new. On the project's test footage
(one camera, small rain-soaked figures) the right identity was the top match
about two times in three - useful for "probably the same person", not for
anything that needs to be certain.

Appearance vectors of people are personal data. Off by default; the gallery
file is written with owner-only permissions and identities expire.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from computer_vision.core.context import FrameContext, Severity
from computer_vision.inference.base import BackendError
from computer_vision.modules.analysis.common import (
    SubjectMemory,
    make_event,
    parse_severity,
    positive_number,
)
from computer_vision.modules.analysis.geometry import subject_key
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_MODEL = "computer_vision/Model/reid/person_reid_youtu_2021nov.onnx"
EVENT_REIDENTIFIED = "person_reidentified"

Embed = Callable[[np.ndarray], np.ndarray]

#: Cameras of one multi-camera run sync the same gallery file from different
#: threads. The file is replaced atomically, which on Windows fails if another
#: thread has it open - so read and write it one thread at a time.
_FILE_LOCK = threading.Lock()


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / (float(np.linalg.norm(vector)) + 1e-9)


@dataclass
class Identity:
    global_id: str
    embedding: np.ndarray
    last_seen: float
    source_id: str
    sightings: int = 1


class Gallery:
    """Known identities, optionally mirrored to a file other cameras share."""

    def __init__(self, path: str | None = None, ttl_seconds: float = 1800.0) -> None:
        self.path = Path(path) if path else None
        self.ttl_seconds = ttl_seconds
        self.identities: Dict[str, Identity] = {}

    def best_match(self, embedding: np.ndarray, exclude: set[str]) -> Tuple[Identity | None, float]:
        best: Identity | None = None
        best_score = -1.0
        for identity in self.identities.values():
            if identity.global_id in exclude:
                continue
            score = float(np.dot(embedding, identity.embedding))
            if score > best_score:
                best, best_score = identity, score
        return best, best_score

    def enrol(self, embedding: np.ndarray, now: float, source_id: str) -> Identity:
        identity = Identity(uuid.uuid4().hex[:8], _unit(embedding), now, source_id)
        self.identities[identity.global_id] = identity
        return identity

    def confirm(self, identity: Identity, embedding: np.ndarray, now: float, source_id: str) -> None:
        """Fold a new sighting into an identity, so it tracks slow change."""
        weight = min(identity.sightings, 9)
        identity.embedding = _unit(identity.embedding * weight + embedding)
        identity.sightings += 1
        identity.last_seen = now
        identity.source_id = source_id

    def prune(self, now: float) -> None:
        if self.ttl_seconds <= 0:
            return
        for global_id in [
            gid for gid, item in self.identities.items() if now - item.last_seen > self.ttl_seconds
        ]:
            del self.identities[global_id]

    # ---- shared file ---- #

    def load(self) -> None:
        """Merge the shared file in. For each identity the newer sighting wins."""
        if self.path is None or not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                ids = [str(v) for v in data["ids"]]
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                last_seen = np.asarray(data["last_seen"], dtype=np.float64)
                sources = [str(v) for v in data["sources"]]
                sightings = np.asarray(data["sightings"], dtype=np.int64)
        except (OSError, ValueError, KeyError):
            return  # a half-written or foreign file: keep what is in memory

        if not (len(ids) == len(embeddings) == len(last_seen) == len(sources) == len(sightings)):
            return
        for index, global_id in enumerate(ids):
            if not np.all(np.isfinite(embeddings[index])):
                continue
            known = self.identities.get(global_id)
            if known is None or last_seen[index] > known.last_seen:
                self.identities[global_id] = Identity(
                    global_id, _unit(embeddings[index]), float(last_seen[index]),
                    sources[index], int(sightings[index]),
                )

    def save(self) -> None:
        """Write atomically: other cameras must never read half a gallery."""
        if self.path is None:
            return
        items = list(self.identities.values())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as handle:
                np.savez(
                    handle,
                    ids=np.array([item.global_id for item in items], dtype=np.str_),
                    embeddings=(
                        np.stack([item.embedding for item in items])
                        if items else np.zeros((0, 1), dtype=np.float32)
                    ),
                    last_seen=np.array([item.last_seen for item in items], dtype=np.float64),
                    sources=np.array([item.source_id for item in items], dtype=np.str_),
                    sightings=np.array([item.sightings for item in items], dtype=np.int64),
                )
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass
class _Person:
    samples: List[np.ndarray] = field(default_factory=list)
    last_sample_frame: int = -(10**9)
    global_id: str = ""


class PersonReIdModule(BaseAIModule):
    """Gives tracked people an appearance-based id that survives re-entry.

    Config block::

        person_reid:
          enabled: true
          model: computer_vision/Model/reid/person_reid_youtu_2021nov.onnx   # you supply this
          device: cpu
          threshold: 0.6           # cosine similarity to call two people the same
          min_height: 100          # px in the original capture; smaller crops are skipped
          samples: 3               # crops averaged before deciding
          interval_frames: 5       # frames between crops of one person
          max_per_frame: 2         # crops embedded per frame (the model is heavy)
          identity_ttl_seconds: 1800
          gallery_path: null       # shared file; set the same path on every camera
          severity: info

    Each track it has decided on carries ``extra["global_id"]``.
    Requires ``person_tracking`` and the ``openvino`` package.
    """

    module_name = "person_reid"
    module_type = ModuleType.RECOGNITION
    version = "1.0.0"
    requires = ("person_tracking",)

    def initialize(self) -> None:
        self._threshold = positive_number(self.params, "threshold", 0.6)
        if self._threshold > 1.0:
            raise ModuleInitializationError("person_reid.threshold is a cosine similarity (0-1)")
        self._min_height = positive_number(self.params, "min_height", 100.0, allow_zero=True)
        self._samples = max(1, int(self.params.get("samples", 3)))
        self._interval = max(1, int(self.params.get("interval_frames", 5)))
        self._max_per_frame = max(1, int(self.params.get("max_per_frame", 2)))
        self._sync_seconds = positive_number(self.params, "sync_seconds", 2.0)
        self._severity = parse_severity(self.params, Severity.INFO)

        try:
            self._embed: Embed = self._build_embedder()
        except BackendError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        self._gallery = Gallery(
            self.params.get("gallery_path"),
            positive_number(self.params, "identity_ttl_seconds", 1800.0, allow_zero=True),
        )
        self._gallery.load()
        self._memory: SubjectMemory[_Person] = SubjectMemory(forget_seconds=5.0)
        self._last_sync = 0.0
        self._total = 0
        self.log.info(
            "Ready: similarity >= %.2f, %d sample(s) per person, gallery=%s (%d known)",
            self._threshold, self._samples,
            self._gallery.path or "in memory", len(self._gallery.identities),
        )

    def _build_embedder(self) -> Embed:
        """Load the network. A seam so tests need no model file."""
        from computer_vision.inference.person_embedder import PersonEmbedder

        return PersonEmbedder(
            str(self.params.get("model") or DEFAULT_MODEL),
            sha256=self.params.get("model_sha256"),
            device=str(self.params.get("device", "cpu")),
        ).embed

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        wall = time.time()
        image = context.original_frame if context.original_frame is not None else context.frame
        scale_x = image.shape[1] / context.width
        scale_y = image.shape[0] / context.height

        tracks = context.get_tracks("person_tracking")
        events = []
        embedded = 0

        # Ids already worn by someone in view: one person cannot be two tracks
        # of the same camera at once, so those are not candidates.
        in_view = set()
        for track in tracks:
            state = self._memory.get(subject_key(track.source_module, track.track_id))
            if state is not None and state.global_id:
                in_view.add(state.global_id)

        for track in sorted(tracks, key=lambda t: t.bbox.area, reverse=True):
            key = subject_key(track.source_module, track.track_id)
            state = self._memory.get(key) or _Person()
            self._memory.put(key, state, now)

            if state.global_id:
                track.extra["global_id"] = state.global_id
                known = self._gallery.identities.get(state.global_id)
                if known is not None:
                    known.last_seen = wall  # still here: keep it from expiring
                continue

            if embedded >= self._max_per_frame:
                continue
            if context.frame_id - state.last_sample_frame < self._interval:
                continue
            crop = self._crop(track, image, scale_x, scale_y)
            if crop is None:
                continue

            embedded += 1
            state.last_sample_frame = context.frame_id
            state.samples.append(_unit(self._embed(crop)))
            if len(state.samples) < self._samples:
                continue

            appearance = _unit(np.mean(state.samples, axis=0))
            match, score = self._gallery.best_match(appearance, exclude=in_view)
            if match is not None and score >= self._threshold:
                previous_source, unseen = match.source_id, wall - match.last_seen
                self._gallery.confirm(match, appearance, wall, context.source_id)
                state.global_id = match.global_id
                events.append(
                    make_event(
                        self.module_name, EVENT_REIDENTIFIED, context, self._severity,
                        subject={"track_id": track.track_id, "class": track.label, "identity": match.global_id},
                        data={
                            "global_id": match.global_id,
                            "similarity": round(score, 3),
                            "previous_source": previous_source,
                            "other_camera": previous_source != context.source_id,
                            "unseen_seconds": round(max(0.0, unseen), 1),
                        },
                    )
                )
            else:
                state.global_id = self._gallery.enrol(appearance, wall, context.source_id).global_id

            state.samples.clear()
            in_view.add(state.global_id)
            track.extra["global_id"] = state.global_id

        self._memory.expire(now)
        self._sync(wall)

        for event in events:
            context.add_event(event)
            self.log.info(
                "%s -> %s (%.2f)", event.describe(), event.data["global_id"], event.data["similarity"]
            )
        self._total += len(events)

        return self.make_result(
            context,
            data={
                "identities_known": len(self._gallery.identities),
                "identified_in_view": len(in_view),
                "embedded": embedded,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def _crop(self, track, image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray | None:
        x1 = int(max(0, track.bbox.x1 * scale_x))
        y1 = int(max(0, track.bbox.y1 * scale_y))
        x2 = int(min(image.shape[1], track.bbox.x2 * scale_x))
        y2 = int(min(image.shape[0], track.bbox.y2 * scale_y))
        if y2 - y1 < max(self._min_height, 8) or x2 - x1 < 8:
            return None
        return image[y1:y2, x1:x2]

    def _sync(self, wall: float, force: bool = False) -> None:
        """Exchange identities with the other cameras through the shared file."""
        self._gallery.prune(wall)
        if self._gallery.path is None:
            return
        if not force and wall - self._last_sync < self._sync_seconds:
            return
        self._last_sync = wall
        try:
            with _FILE_LOCK:
                self._gallery.load()
                self._gallery.prune(wall)
                self._gallery.save()
        except OSError as exc:
            self.log.warning("Could not sync the re-identification gallery: %s", exc)

    @property
    def gallery(self) -> Gallery:
        return self._gallery

    def cleanup(self) -> None:
        if hasattr(self, "_gallery"):
            self._sync(time.time(), force=True)
        if hasattr(self, "_memory"):
            self._memory.clear()
        self._embed = None  # type: ignore[assignment]
        super().cleanup()
