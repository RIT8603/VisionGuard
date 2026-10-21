"""Face Recognition module.

One responsibility: decide **"whose face is this?"** for faces that
``face_detection`` already found.

It does not detect faces, does not perform attendance, does not raise alerts
and does not enrol anyone. An identity exists in the store because a human put
it there deliberately - nothing here ever adds one from a passing face
(Agent.md §14, §24).

Enabling this module is the point at which VisionGuard starts producing
personal data. It is off by default.

Two sources of identities:

* ``identity_store: <file.npz>`` - a fixed file read once at start-up.
* no ``identity_store`` - the **live shared store** that the dashboard's face
  registry writes to. Users registered in the UI are recognised on the next
  frame, with no restart.

How a name is decided (accuracy comes from *when* the recogniser runs, not from
running it more):

1. **Quality gate.** A face that is small, blurred or turned too far is never
   embedded - such a vector looks confident and points at the wrong person.
2. **Subjects.** Each face is attached to a *subject*. With ``person_tracking``
   enabled that is the person track whose body box contains the face, so the
   name stays with the person while they turn away, walk off or go side-on and
   the face detector loses them. Faces are also followed on their own by a
   FindFace-style tracker (box overlap with velocity prediction), so a face
   keeps its subject in frames where it is not inside any person box.
3. **Best snapshot.** A subject is recognised as soon as a good-enough face
   appears, then again only when a clearly *better* one arrives (closer,
   sharper, more frontal). Otherwise the known name is kept - no flicker, and
   far less work than embedding every face on every frame.
4. **Strict match.** A match must clear the similarity threshold *and* beat the
   runner-up identity by ``margin``; two look-alikes stay "unknown" rather than
   being guessed. A name is confirmed only after enough agreeing matches, and
   replaced only by repeated, better evidence of someone else.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.identity.quality import FaceQuality, assess_face
from computer_vision.identity.shared import get_shared_store
from computer_vision.identity.store import IdentityStore, IdentityStoreError
from computer_vision.inference.face_base import BackendError, FaceEmbedderBackend
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType
from computer_vision.modules.recognition.face_tracker import FaceTracker

DEFAULT_BACKEND = "sface"
UNKNOWN_LABEL = "unknown"
FACE_LABEL = "face"
SPOOF_LABEL = "spoof"
PERSON_TRACKER = "person_tracking"

#: A person-track subject is forgotten after this many frames unseen. Long on
#: purpose: the body tracker outlives the face detector, which is the point.
_PERSON_SUBJECT_MAX_AGE = 120
#: Share of the face box that must lie inside a person box to belong to it.
_CONTAINMENT = 0.8
#: A face sits in the upper part of a body: its centre must be above this
#: fraction of the person box's height.
_HEAD_REGION = 0.6
#: A snapshot this much better than the one a name was based on triggers a re-check.
_IMPROVEMENT = 0.08
#: Face quality at which a single match is trusted at once.
_HIGH_QUALITY = 0.80


class _Subject:
    """What is known about one person (or one followed face)."""

    __slots__ = (
        "face_bbox", "age", "name", "similarity", "best_quality",
        "candidate", "candidate_hits", "last_checked", "checked",
    )

    def __init__(self, face_bbox: BBox) -> None:
        self.face_bbox = face_bbox
        self.age = 0
        self.name: Optional[str] = None
        self.similarity = 0.0
        #: Quality of the best snapshot the current name rests on.
        self.best_quality = 0.0
        #: An identity being considered but not yet confirmed (or a challenger).
        self.candidate: Optional[str] = None
        self.candidate_hits = 0
        self.last_checked = -10**9
        #: A good-enough face has been compared against the store at least once.
        #: Until then "unknown" would be a guess, not a finding.
        self.checked = False


def _inside(face: BBox, person: BBox) -> bool:
    """Whether ``face`` belongs to the body ``person``: contained, in the head region."""
    area = face.width * face.height
    if area <= 0:
        return False
    w = max(0.0, min(face.x2, person.x2) - max(face.x1, person.x1))
    h = max(0.0, min(face.y2, person.y2) - max(face.y1, person.y1))
    if (w * h) / area < _CONTAINMENT:
        return False
    centre_y = (face.y1 + face.y2) / 2.0
    return centre_y <= person.y1 + _HEAD_REGION * person.height


class FaceRecognitionModule(BaseAIModule):
    """Matches detected faces against enrolled identities.

    Config block::

        face_recognition:
          enabled: true
          backend: sface
          model: computer_vision/Model/face/face_recognition_sface_2021dec.onnx   # required
          identity_store: computer_vision/Model/face/identities.npz               # optional
          threshold: 0.38         # cosine similarity needed to name a face
          margin: 0.05            # best identity must beat the runner-up by this
          min_face_size: 60       # smaller faces are not recognised (px)
          max_angle: 45           # yaw or pitch beyond this (deg) is not recognised
          min_sharpness: 20       # blurrier faces are not recognised
          min_quality: 0.42       # combined 0..1 score below which a face is skipped
          confirm_matches: 2      # agreeing matches before a name is trusted
          embed_interval: 2       # frames between recognitions of one subject
          max_faces: 10           # cap faces examined per frame
          track_overlap_threshold: 0.25  # face tracker: IoU to continue a track
          track_miss_interval: 1.0       # face tracker: seconds unseen before it ends

    Leave ``identity_store`` out to use the live registry managed from the
    dashboard (Face Registry page). Enable ``person_tracking`` as well and the
    name follows the person's body, not just the face.

    Publishes one detection per face labelled with the identity, or
    ``unknown``. Boxes are not drawn twice: the source detector is marked
    consumed, exactly as tracking does.
    """

    module_name = "face_recognition"
    module_type = ModuleType.RECOGNITION
    version = "2.0.0"
    requires = ("face_detection",)
    #: Use this frame's person tracks when the tracker is on.
    run_after = (PERSON_TRACKER, "face_liveness")

    source_module = "face_detection"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._embedder: FaceEmbedderBackend | None = None
        self._store: IdentityStore | None = None
        self._max_faces = int(self.params.get("max_faces", 10))
        self._threshold = self.params.get("threshold")
        self._margin = float(self.params.get("margin", 0.05))
        self._min_size = float(self.params.get("min_face_size", 60))
        self._max_angle = float(self.params.get("max_angle", 45.0))
        self._min_sharp = float(self.params.get("min_sharpness", 20))
        self._min_quality = float(self.params.get("min_quality", 0.42))
        self._confirm = max(1, int(self.params.get("confirm_matches", 2)))
        self._interval = max(1, int(self.params.get("embed_interval", 2)))
        self._by_person: Dict[int, _Subject] = {}
        #: Face track id -> the subject that face currently belongs to (its own,
        #: or its person's), so a name survives the face leaving a body box.
        self._by_face: Dict[int, _Subject] = {}
        self._faces = FaceTracker(
            overlap_threshold=float(self.params.get("track_overlap_threshold", 0.25)),
            miss_interval=float(self.params.get("track_miss_interval", 1.0)),
        )
        self._live = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        from computer_vision.inference.face_factory import create_face_embedder

        store_path = self.params.get("identity_store")

        backend_name = str(self.params.get("backend", DEFAULT_BACKEND))
        try:
            self._embedder = create_face_embedder(backend_name, self.params)
            self._embedder.load()
        except BackendError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        threshold = (
            float(self._threshold)
            if self._threshold is not None
            else self._embedder.default_threshold
        )
        if store_path:
            try:
                self._store = IdentityStore.load(store_path, threshold=threshold)
            except IdentityStoreError as exc:
                raise ModuleInitializationError(str(exc)) from exc
            self._live = False
        else:
            # The live registry: shared with the dashboard, updated while running.
            self._store = get_shared_store()
            self._store.threshold = threshold
            self._live = True

        if self._store.is_empty():
            self.log.warning(
                "The identity store has no identities - every face will be reported as "
                "unknown until users are registered (Face Registry page)."
            )

        self.log.info(
            "Ready: backend=%s, %d identities (%d samples), threshold=%.3f, margin=%.2f, "
            "min_face=%.0fpx",
            self._embedder.description,
            len(self._store),
            self._store.sample_count,
            threshold,
            self._margin,
            self._min_size,
        )

    def cleanup(self) -> None:
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None
        # Drop the embeddings from memory promptly rather than waiting for GC.
        # The live shared store belongs to the registry, not to this module.
        self._store = None
        self._by_person.clear()
        self._by_face.clear()
        self._faces.reset()
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        if self._embedder is None or self._store is None:
            raise RuntimeError(
                "FaceRecognitionModule.process() called before initialize()"
            )

        faces = context.get_detections(self.source_module)[: self._max_faces]
        persons = context.get_tracks(PERSON_TRACKER)
        results: List[Detection] = []
        matched = 0

        for subject in self._by_person.values():
            subject.age += 1

        face_ids, closed = self._faces.update(
            [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2) for d in faces],
            context.frame_id,
            context.monotonic_ts,
        )
        for face_id in closed:
            self._by_face.pop(face_id, None)

        for face_detection, face_id in zip(faces, face_ids):
            raw_face = face_detection.extra.get("face")
            if raw_face is None:
                continue

            subject, person_id = self._subject_for(face_detection.bbox, persons, face_id)

            # Liveness gate: a face judged a photo/screen is never named.
            liveness = face_detection.extra.get("liveness") or {}
            if liveness.get("state") == "spoof" and liveness.get("block", True):
                results.append(
                    Detection(
                        label=SPOOF_LABEL,
                        confidence=face_detection.confidence,
                        bbox=face_detection.bbox,
                        source_module=self.module_name,
                        extra={"identity": None, "recognised": False, "verified": False,
                               "liveness": "spoof", "face_track_id": face_id},
                    )
                )
                continue
            quality = assess_face(
                context.frame,
                raw_face.xyxy,
                raw_face.landmarks,
                face_detection.confidence,
                min_size=self._min_size,
                max_angle=self._max_angle,
                min_sharpness=self._min_sharp,
                min_score=self._min_quality,
            )

            similarity = subject.similarity
            if quality.usable and self._should_check(subject, quality, context.frame_id):
                subject.last_checked = context.frame_id
                try:
                    embedding = self._embedder.embed(context.frame, raw_face)
                except Exception as exc:  # noqa: BLE001 - one bad crop is not fatal
                    self.log.debug("Could not embed a face: %s", exc)
                    if subject.name is None:
                        continue  # nothing to say about this face this frame
                else:
                    match = self._store.match(embedding, margin=self._margin)
                    similarity = match.similarity
                    subject.checked = True
                    self._observe(subject, match.name if match.matched else None, match.similarity, quality)

            is_known = subject.name is not None
            matched += int(is_known)

            # "unknown" is a finding only once a usable face has been compared;
            # a face that was too small, blurred or turned is just "face".
            verified = is_known or subject.checked
            extra: Dict[str, Any] = {
                "identity": subject.name,
                "similarity": round(similarity, 4),
                "recognised": is_known,
                "verified": verified,
                "quality": quality.score,
            }
            if person_id is not None:
                extra["person_track_id"] = person_id
            if face_id >= 0:
                extra["face_track_id"] = face_id
            if not quality.usable:
                extra["skipped"] = list(quality.problems)
            if liveness:
                extra["liveness"] = liveness.get("state")

            results.append(
                Detection(
                    label=subject.name if is_known else (UNKNOWN_LABEL if verified else FACE_LABEL),
                    confidence=face_detection.confidence,
                    bbox=face_detection.bbox,
                    source_module=self.module_name,
                    extra=extra,
                )
            )

        self._by_person = {
            pid: s for pid, s in self._by_person.items() if s.age <= _PERSON_SUBJECT_MAX_AGE
        }

        context.add_detections(self.module_name, results)

        return self.make_result(
            context,
            detections=[det.to_dict() for det in results],
            data={
                "count": len(results),
                "recognised": matched,
                "unknown": len(results) - matched,
                "enrolled_identities": len(self._store),
                "consumed_from": self.source_module,
                "following": "person_tracks" if persons else "face_tracks",
            },
        )

    # ------------------------------------------------------------------ #
    # Subjects
    # ------------------------------------------------------------------ #

    def _subject_for(
        self, face: BBox, persons: Sequence[Track], face_id: int
    ) -> tuple[_Subject, Optional[int]]:
        """The subject this face belongs to, and the person track id if any.

        A face inside a body box belongs to that person track. Otherwise it
        belongs to its own face track. The two are kept in step, so a name
        learned through either survives the other being lost: the face track
        remembers the person's subject when the body box drops out, and a body
        that is picked up late adopts the name its face already had.
        """
        owner: Optional[Track] = None
        for track in persons:
            if _inside(face, track.bbox):
                # Several bodies can contain a face (one behind another); the
                # smallest is the one the face is actually on.
                if owner is None or track.bbox.width * track.bbox.height < owner.bbox.width * owner.bbox.height:
                    owner = track

        known = self._by_face.get(face_id) if face_id >= 0 else None

        if owner is not None:
            subject = self._by_person.get(owner.track_id)
            if subject is None or (subject.name is None and known is not None and known.name is not None):
                subject = known if known is not None else _Subject(face)
                self._by_person[owner.track_id] = subject
            person_id: Optional[int] = owner.track_id
        else:
            subject = known if known is not None else _Subject(face)
            person_id = None

        subject.face_bbox, subject.age = face, 0
        if face_id >= 0:
            self._by_face[face_id] = subject
        return subject, person_id

    # ------------------------------------------------------------------ #
    # Deciding whether to recognise, and what to believe
    # ------------------------------------------------------------------ #

    def _should_check(self, subject: _Subject, quality: FaceQuality, frame_id: int) -> bool:
        """Run the recogniser only when it can add something.

        An unnamed subject is tried every ``embed_interval`` frames. A named
        one is only re-checked when a clearly better snapshot has arrived;
        otherwise the known name simply stays.
        """
        if frame_id - subject.last_checked < self._interval:
            return False
        if subject.name is None:
            return True
        return quality.score >= subject.best_quality + _IMPROVEMENT

    def _observe(
        self, subject: _Subject, name: Optional[str], similarity: float, quality: FaceQuality
    ) -> None:
        """Fold one recognition result into what the subject is believed to be."""
        needed = 1 if quality.score >= _HIGH_QUALITY else self._confirm

        if name is None:
            # "Unknown" is never allowed to erase a name (the person may just
            # have turned), and does not count against an unconfirmed one.
            subject.candidate, subject.candidate_hits = None, 0
            return

        if subject.name == name:
            subject.similarity = similarity
            subject.best_quality = max(subject.best_quality, quality.score)
            subject.candidate, subject.candidate_hits = None, 0
            return

        # A name that is new, or someone else than the current one.
        if subject.candidate == name:
            subject.candidate_hits += 1
        else:
            subject.candidate, subject.candidate_hits = name, 1

        # Replacing an existing name takes more evidence than giving the first one.
        required = needed if subject.name is None else max(needed, 2)
        if subject.candidate_hits >= required:
            subject.name = name
            subject.similarity = similarity
            subject.best_quality = quality.score
            subject.candidate, subject.candidate_hits = None, 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def enrolled_names(self) -> List[str]:
        return self._store.names if self._store else []
