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
   being guessed. A name is confirmed only after ``confirm_matches`` agreeing
   matches - always, however sharp the face.
5. **Revalidation.** A confirmed name is checked again every
   ``revalidate_seconds`` while the face is visible. ``confirm_matches``
   disagreeing checks in a row (someone else, or a face that no longer
   resembles the named person) withdraw the name. Changing the registry
   (a user added, edited, deleted or re-enrolled) re-checks every name.
6. **Body continuity.** A confirmed name stays with the person track while the
   face is turned away, refreshed as long as that track is reliable. If the
   body was lost for a moment or overlapped someone else, the name is held
   back ("checking") until the face is seen and matches again.

Every face is published with a ``state``: ``recognised`` (a confirmed name),
``unknown`` (compared and matched nobody), ``checking`` (evidence is still
being collected or a name is being re-verified) or ``unclear`` (the face has
not yet been good enough to compare).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from computer_vision.core.context import BBox, Detection, FrameContext, Track
from computer_vision.identity.quality import FaceQuality, assess_face
from computer_vision.identity.shared import get_shared_store, store_version
from computer_vision.identity.store import IdentityStore, IdentityStoreError
from computer_vision.inference.face_base import BackendError, FaceEmbedderBackend, RawFace
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType
from computer_vision.modules.recognition.face_tracker import FaceTracker

DEFAULT_BACKEND = "sface"
UNKNOWN_LABEL = "unknown"
#: Label of a face that has not been good enough to compare yet.
FACE_LABEL = "face"
CHECKING_LABEL = "checking"
SPOOF_LABEL = "spoof"
PERSON_TRACKER = "person_tracking"

#: ``extra["state"]`` values.
STATE_RECOGNISED = "recognised"
STATE_UNKNOWN = "unknown"
STATE_CHECKING = "checking"
STATE_UNCLEAR = "unclear"
#: ``context.metadata`` key: person track id -> {"name", "state"} for every
#: person this module has an opinion on, face visible or not.
PERSON_IDENTITIES_KEY = "person_identities"
#: Detection label for a face without a confirmed name, by state.
_LABELS = {STATE_UNKNOWN: UNKNOWN_LABEL, STATE_CHECKING: CHECKING_LABEL, STATE_UNCLEAR: FACE_LABEL}

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
#: Small-face embeddings averaged per subject (about a second of checks).
_SMALL_WINDOW = 10


class _Subject:
    """What is known about one person (or one followed face)."""

    __slots__ = (
        "face_bbox", "age", "name", "similarity", "best_quality",
        "candidate", "candidate_hits", "last_checked", "checked", "recent",
        "verified_at", "disagree", "disagree_name", "uncertain", "last_decision",
        "body_seen_at",
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
        #: Latest small-face embeddings (unit length), averaged before matching.
        self.recent: Deque[np.ndarray] = deque(maxlen=_SMALL_WINDOW)
        #: Clock time the current name was last confirmed by a face.
        self.verified_at = float("-inf")
        #: Consecutive checks that disagreed with the current name, and the
        #: name they pointed at (None when they matched nobody or disagreed).
        self.disagree = 0
        self.disagree_name: Optional[str] = None
        #: The name must be re-verified before it is shown again (the body was
        #: lost for a moment, overlapped someone, or the registry changed).
        self.uncertain = False
        #: Outcome of the latest decisive comparison: "match", "nomatch" or None.
        self.last_decision: Optional[str] = None
        #: Clock time this subject's person track was last observed.
        self.body_seen_at: Optional[float] = None


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
          small_face_size: 60     # below this a face needs...
          small_face_threshold: 0     # ...this similarity (0 = threshold), and
          small_face_confirm: 3       # ...this many views averaged first
          small_face_margin: 0.08     # ...each beating the runner-up by this
          max_angle: 45           # yaw or pitch beyond this (deg) is not recognised
          min_sharpness: 20       # blurrier faces are not recognised
          min_quality: 0.42       # combined 0..1 score below which a face is skipped
          confirm_matches: 2      # agreeing matches to give a name, disagreeing ones to withdraw it
          revalidate_seconds: 3   # re-check a confirmed name this often while the face is visible
          body_overlap_iou: 0.3   # body overlapping another person this much: re-verify the name
          body_gap_reverify_seconds: 0.5  # body unseen longer than this: re-verify the name
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
        # CCTV faces are often 20-60 px. There one comparison is weak evidence
        # (a stranger can score 0.45 once). So a small face is matched by the
        # average of the subject's last few embeddings, only once there are
        # small_face_confirm of them, with a wider margin, and the average
        # must agree twice in a row before the name is shown.
        self._small_size = float(self.params.get("small_face_size", 60))
        self._small_threshold = float(self.params.get("small_face_threshold", 0.0))
        self._small_confirm = max(2, int(self.params.get("small_face_confirm", 3)))
        # Measured on real stairs CCTV: genuine small-face checks beat the
        # runner-up by 0.08-0.20, a stranger's lucky matches by <= 0.076.
        self._small_margin = float(self.params.get("small_face_margin", 0.08))
        self._interval = max(1, int(self.params.get("embed_interval", 2)))
        # A confirmed name is re-checked this often while the face is visible,
        # so a wrong one does not stick for as long as the person stays.
        self._revalidate = float(self.params.get("revalidate_seconds", 3.0))
        # Body continuity: overlap with another person's box at which the body
        # can no longer be told apart, and a gap after which a returning track
        # may be someone else - either way the name needs a face again.
        self._body_overlap = float(self.params.get("body_overlap_iou", 0.3))
        self._body_gap = float(self.params.get("body_gap_reverify_seconds", 0.5))
        self._store_version: Optional[int] = None
        self._by_person: Dict[int, _Subject] = {}
        #: Face track id -> the subject that face currently belongs to (its own,
        #: or its person's), so a name survives the face leaving a body box.
        self._by_face: Dict[int, _Subject] = {}
        self._faces = FaceTracker(
            overlap_threshold=float(self.params.get("track_overlap_threshold", 0.25)),
            miss_interval=float(self.params.get("track_miss_interval", 1.0)),
            center_gate=float(self.params.get("track_center_gate", 0.0)),
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

        now = context.monotonic_ts
        self._sync_store()
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
                               "state": "spoof", "liveness": "spoof", "face_track_id": face_id},
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
            collecting = False
            if quality.usable and self._should_check(subject, quality, context.frame_id, now):
                subject.last_checked = context.frame_id
                try:
                    embedding = self._embed(context, raw_face)
                except Exception as exc:  # noqa: BLE001 - one bad crop is not fatal
                    self.log.debug("Could not embed a face: %s", exc)
                    if subject.name is None:
                        continue  # nothing to say about this face this frame
                else:
                    small = quality.size_px < self._small_size
                    decide = True
                    if small:
                        # One small face is a weak, noisy view; the average of
                        # the subject's recent views is not. Strangers' chance
                        # matches do not survive averaging, a real match does.
                        vector = np.asarray(embedding, dtype=np.float32).ravel()
                        subject.recent.append(vector / (np.linalg.norm(vector) or 1.0))
                        if len(subject.recent) < self._small_confirm:
                            decide = False
                            collecting = True
                        else:
                            mean = np.mean(subject.recent, axis=0)
                            embedding = mean / (np.linalg.norm(mean) or 1.0)
                    if decide:
                        margin = max(self._margin, self._small_margin) if small else self._margin
                        match = self._store.match(embedding, margin=margin)
                        similarity = match.similarity
                        subject.checked = True
                        name = match.name if match.matched else None
                        if small and similarity < self._small_threshold:
                            name = None
                        self._observe(subject, name, match.similarity, quality, now, embedding, small=small)

            state = self._state_of(subject, quality.usable, collecting)
            is_known = state == STATE_RECOGNISED
            matched += int(is_known)
            extra: Dict[str, Any] = {
                "identity": subject.name if is_known else None,
                "similarity": round(similarity, 4),
                "recognised": is_known,
                # A finding (a name, or "nobody we know") - as opposed to a
                # face still being checked or too unclear to compare.
                "verified": state in (STATE_RECOGNISED, STATE_UNKNOWN),
                "state": state,
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
                    label=str(subject.name) if is_known else _LABELS[state],
                    confidence=face_detection.confidence,
                    bbox=face_detection.bbox,
                    source_module=self.module_name,
                    extra=extra,
                )
            )

        self._follow_bodies(context, persons, now)
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
                "unknown": sum(1 for d in results if d.extra.get("state") == STATE_UNKNOWN),
                "checking": sum(1 for d in results if d.extra.get("state") == STATE_CHECKING),
                "unclear": sum(1 for d in results if d.extra.get("state") == STATE_UNCLEAR),
                "enrolled_identities": len(self._store),
                "consumed_from": self.source_module,
                "following": "person_tracks" if persons else "face_tracks",
            },
        )

    # ------------------------------------------------------------------ #
    # States, registry changes and body continuity
    # ------------------------------------------------------------------ #

    @staticmethod
    def _state_of(subject: _Subject, usable: bool, collecting: bool) -> str:
        """What can honestly be said about this subject right now."""
        if subject.name is not None and not subject.uncertain:
            return STATE_RECOGNISED
        if subject.name is not None or subject.candidate is not None or collecting:
            return STATE_CHECKING  # a name to (re)confirm, or evidence still coming in
        if subject.last_decision == "nomatch":
            return STATE_UNKNOWN
        return STATE_CHECKING if usable else STATE_UNCLEAR

    def _sync_store(self) -> None:
        """Re-check every name when the live registry has changed.

        A deleted or renamed user's name is withdrawn at once; every other
        name must be confirmed by the face again (templates may have been
        re-enrolled), and until then it shows as "checking".
        """
        if not self._live:
            return
        version = store_version()
        if self._store_version is None:
            self._store_version = version
            return
        if version == self._store_version:
            return
        self._store_version = version
        enrolled = set(self._store.names)
        subjects = {id(s): s for s in [*self._by_person.values(), *self._by_face.values()]}
        for subject in subjects.values():
            if subject.name is not None and subject.name not in enrolled:
                subject.name = None
                subject.last_decision = None
            subject.candidate, subject.candidate_hits = None, 0
            subject.disagree, subject.disagree_name = 0, None
            subject.verified_at = float("-inf")
            subject.last_checked = -10**9
            if subject.name is not None:
                subject.uncertain = True
        self.log.info("Registry changed: %d tracked people will be re-checked", len(subjects))

    def _follow_bodies(self, context: FrameContext, persons: Sequence[Track], now: float) -> None:
        """Keep confirmed names on person tracks while the face is out of sight.

        A track observed this frame, clear of every other person's box, keeps
        its subject alive. One that overlaps another person, or comes back
        after more than ``body_gap_reverify_seconds`` unseen, may now be
        someone else: its name is held back until a face confirms it again.
        """
        identities: Dict[int, Dict[str, Any]] = {}
        for track in persons:
            subject = self._by_person.get(track.track_id)
            if subject is None:
                continue
            crowded = any(
                other.track_id != track.track_id
                and track.bbox.iou(other.bbox) >= self._body_overlap
                for other in persons
            )
            returned = (
                subject.body_seen_at is not None and now - subject.body_seen_at > self._body_gap
            )
            if subject.name is not None and (crowded or returned):
                subject.uncertain = True
            subject.body_seen_at = now
            subject.age = 0
            state = self._state_of(subject, usable=False, collecting=False)
            if state == STATE_UNCLEAR:
                continue  # nothing known about this body yet
            name = subject.name if state == STATE_RECOGNISED else None
            identities[track.track_id] = {"name": name, "state": state}
            track.extra["identity"] = name
            track.extra["identity_state"] = state
        if identities:
            context.metadata.setdefault(PERSON_IDENTITIES_KEY, {}).update(identities)

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

    def _embed(self, context: FrameContext, face: RawFace):
        """Embed from the frame as captured when it was downscaled for processing.

        A 30 px face in the 960 px frame is 60 px in a 1920 px original: twice
        the detail for the recogniser, at no extra detection cost.
        """
        original = context.original_frame
        if original is None or original.shape[1] == context.width:
            return self._embedder.embed(context.frame, face)
        sx = original.shape[1] / context.width
        sy = original.shape[0] / context.height
        x1, y1, x2, y2 = face.xyxy
        row = None
        if face.raw_row is not None:
            # YuNet layout: x, y, w, h, then five (x, y) landmarks.
            row = face.raw_row.astype("float32").copy()
            row[0:14:2] *= sx
            row[1:14:2] *= sy
        scaled = RawFace(
            confidence=face.confidence,
            xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
            landmarks=None if face.landmarks is None else face.landmarks * [sx, sy],
            raw_row=row,
        )
        return self._embedder.embed(original, scaled)

    def _should_check(
        self, subject: _Subject, quality: FaceQuality, frame_id: int, now: float
    ) -> bool:
        """Run the recogniser only when it can add something.

        An unnamed subject is tried every ``embed_interval`` frames. A named
        one is re-checked when its name needs re-verifying (body lost or
        crowded, registry changed, a disagreement pending), every
        ``revalidate_seconds``, and whenever a clearly better snapshot arrives.
        """
        if frame_id - subject.last_checked < self._interval:
            return False
        if subject.name is None or subject.uncertain or subject.disagree:
            return True
        if now - subject.verified_at >= self._revalidate:
            return True
        return quality.score >= subject.best_quality + _IMPROVEMENT

    def _observe(
        self,
        subject: _Subject,
        name: Optional[str],
        similarity: float,
        quality: FaceQuality,
        now: float,
        embedding: Any = None,
        small: bool = False,
    ) -> None:
        """Fold one recognition result into what the subject is believed to be.

        Every name needs ``confirm_matches`` agreeing results before it is
        shown, and the same number of disagreeing results in a row withdraws
        it again.
        """
        needed = self._confirm
        if small:
            # Already an average of several views; two agreeing averages in a row.
            needed = max(needed, 2)
        subject.last_decision = "match" if name is not None else "nomatch"

        if subject.name is None:
            if name is None:
                # A full-size face that matches nobody drops an unconfirmed
                # candidate; a small face keeps collecting evidence across
                # weak checks.
                if not small:
                    subject.candidate, subject.candidate_hits = None, 0
                return
            if subject.candidate == name:
                subject.candidate_hits += 1
            else:
                subject.candidate, subject.candidate_hits = name, 1
            if subject.candidate_hits >= needed:
                self._confirm_name(subject, name, similarity, quality, now)
            return

        if name == subject.name:
            self._confirm_name(subject, name, similarity, quality, now)
            return

        # Disagreement with the name being shown: someone else, or nobody.
        if (
            name is None
            and embedding is not None
            and self._store.score_of(embedding, subject.name) >= self._store.threshold
        ):
            # Still resembles the named person; only too close to someone
            # else to decide. Not evidence against the name.
            return
        subject.disagree = subject.disagree + 1
        if subject.disagree == 1:
            subject.disagree_name = name
        elif subject.disagree_name != name:
            subject.disagree_name = None  # disagreeing checks do not agree with each other
        if subject.disagree < needed:
            return

        withdrawn, challenger = subject.name, subject.disagree_name
        self.log.info(
            "Withdrew the name '%s' after %d disagreeing checks%s",
            withdrawn, subject.disagree, f" (now '{challenger}')" if challenger else "",
        )
        subject.name = None
        subject.uncertain = False
        subject.disagree, subject.disagree_name = 0, None
        subject.candidate, subject.candidate_hits = None, 0
        if challenger is not None:
            # The disagreeing checks consistently named one other person:
            # that is the confirmation the new name needs.
            self._confirm_name(subject, challenger, similarity, quality, now)

    @staticmethod
    def _confirm_name(
        subject: _Subject, name: str, similarity: float, quality: FaceQuality, now: float
    ) -> None:
        if subject.name != name:
            subject.best_quality = quality.score
        else:
            subject.best_quality = max(subject.best_quality, quality.score)
        subject.name = name
        subject.similarity = similarity
        subject.verified_at = now
        subject.uncertain = False
        subject.disagree, subject.disagree_name = 0, None
        subject.candidate, subject.candidate_hits = None, 0
        subject.last_decision = "match"

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def enrolled_names(self) -> List[str]:
        return self._store.names if self._store else []
