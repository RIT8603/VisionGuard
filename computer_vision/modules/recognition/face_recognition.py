"""Face Recognition module.

One responsibility: decide **"whose face is this?"** for faces that
``face_detection`` already found.

It does not detect faces, does not perform attendance, does not raise alerts
and does not enrol anyone. An identity exists in the store because a human put
it there deliberately - nothing here ever adds one from a passing face
(Agent.md §14, §24).

Enabling this module is the point at which VisionGuard starts producing
personal data. It is off by default, it requires an identity store to be
configured, and it refuses to start without one rather than silently labelling
everybody "unknown".
"""

from __future__ import annotations

from typing import Any, Dict, List

from computer_vision.core.context import Detection, FrameContext
from computer_vision.identity.store import IdentityStore, IdentityStoreError
from computer_vision.inference.face_base import BackendError, FaceEmbedderBackend
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_BACKEND = "sface"
UNKNOWN_LABEL = "unknown"


class FaceRecognitionModule(BaseAIModule):
    """Matches detected faces against enrolled identities.

    Config block::

        face_recognition:
          enabled: true
          backend: sface
          model: models/face_recognition_sface_2021dec.onnx   # required
          identity_store: models/identities.npz               # required
          threshold: 0.363        # cosine similarity; SFace's documented value
          max_faces: 10           # cap embeddings per frame

    Publishes one detection per face labelled with the matched identity, or
    ``unknown``. Boxes are not drawn twice: the source detector is marked
    consumed, exactly as tracking does.
    """

    module_name = "face_recognition"
    module_type = ModuleType.RECOGNITION
    version = "1.0.0"
    requires = ("face_detection",)

    source_module = "face_detection"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._embedder: FaceEmbedderBackend | None = None
        self._store: IdentityStore | None = None
        self._max_faces = int(self.params.get("max_faces", 10))
        self._threshold = self.params.get("threshold")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        from computer_vision.inference.face_factory import create_face_embedder

        store_path = self.params.get("identity_store")
        if not store_path:
            raise ModuleInitializationError(
                "face_recognition needs an identity store, and there is no "
                "sensible default - recognising against an empty store would "
                "label everyone 'unknown'.\n"
                "Set modules.face_recognition.identity_store to a .npz built "
                "by computer_vision.identity.IdentityStore (see the README, 'Enrolling "
                "identities')."
            )

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
        try:
            self._store = IdentityStore.load(store_path, threshold=threshold)
        except IdentityStoreError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        if self._store.is_empty():
            self.log.warning(
                "Identity store %s has no identities - every face will be "
                "reported as unknown.",
                store_path,
            )

        self.log.info(
            "Ready: backend=%s, %d identities (%d samples), threshold=%.3f",
            self._embedder.description,
            len(self._store),
            self._store.sample_count,
            threshold,
        )

    def cleanup(self) -> None:
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None
        # Drop the embeddings from memory promptly rather than waiting for GC.
        self._store = None
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
        results: List[Detection] = []
        matched = 0

        for face_detection in faces:
            raw_face = face_detection.extra.get("face")
            if raw_face is None:
                continue

            try:
                embedding = self._embedder.embed(context.frame, raw_face)
            except Exception as exc:  # noqa: BLE001 - one bad crop is not fatal
                self.log.debug("Could not embed a face: %s", exc)
                continue

            match = self._store.match(embedding)
            matched += int(match.matched)

            results.append(
                Detection(
                    label=match.label,
                    confidence=face_detection.confidence,
                    bbox=face_detection.bbox,
                    source_module=self.module_name,
                    extra={
                        "identity": match.name if match.matched else None,
                        "similarity": round(match.similarity, 4),
                        "recognised": match.matched,
                    },
                )
            )

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
            },
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def enrolled_names(self) -> List[str]:
        return self._store.names if self._store else []
