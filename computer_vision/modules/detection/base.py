"""Shared machinery for label-filtered detection modules.

Almost every detector in VisionGuard does the same three things: run a model,
keep the labels it cares about, and publish them under its own module name.
The differences are *which* labels and *which* weights - both configuration,
not code.

This base holds that shared behaviour. It is not a merging of modules: each
subclass keeps its own ``module_name``, its own config block, its own
enable/disable switch and its own namespace in the ``FrameContext``. A module
still does one job; it just does not re-implement the plumbing.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, List, Sequence, Tuple  # noqa: F401

from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference.base import BackendError, DetectorBackend, RawDetection
from computer_vision.inference.factory import create_detector
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_BACKEND = "ultralytics"
DEFAULT_COCO_MODEL = "computer_vision/Model/detection/yolo11n.pt"

#: Class ids in the COCO dataset that the stock models are trained on.
COCO_CLASS_IDS: Dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
}


class LabelFilteredDetectionModule(BaseAIModule):
    """Runs a detector backend and publishes the labels this module owns.

    Subclasses declare:
        module_name         - the config key and result namespace
        default_labels      - which classes this module reports
        default_model       - weights to use when config does not say
        requires_own_model  - True when no stock model contains these classes

    Config keys common to all subclasses::

        <module_name>:
          enabled: true
          backend: ultralytics
          model: yolo11n.pt
          device: cpu
          imgsz: 640
          confidence_threshold: 0.4
          min_box_area: 0
          candidate_threshold: 0.25  # optional, see below
          labels: [...]        # override the reported classes
          exclude_labels: [...] # classes never reported (after `labels`)
          dedupe_iou: 0.7      # one box per place across classes; 0 = off
          share_model: true    # reuse a matching model across modules

    Class selection is done here, by label, rather than by restricting the
    model to class ids. Measured, restricting a YOLO model to specific classes
    saves nothing (+0.2%, within noise) because the filter runs after NMS - and
    filtering here instead lets several modules share one inference.

    ``exclude_labels`` exists for the general-purpose detector: with no
    ``labels`` it reports everything, people and vehicles included, so
    running it beside ``person_detection`` puts two boxes (and two track ids)
    on every person. Excluding ``person`` there leaves one.

    ``dedupe_iou`` handles the other kind of duplicate. The model's own NMS
    works per class, so one pile of clutter can come back as a "suitcase"
    *and* a "chair" in the same place; with this set, boxes of different
    classes overlapping at least that much are reduced to the more confident
    one.

    ``candidate_threshold`` (off unless set) additionally hands boxes scoring
    between it and ``confidence_threshold`` to the tracker as *candidates*.
    They are not detections - nothing reports, draws or counts them - but a
    tracker may use one to continue an identity it already has. The backend
    itself drops everything below ``model_confidence`` (default 0.25), so a
    lower value here has no effect unless that is lowered too.
    """

    module_type: ClassVar[ModuleType] = ModuleType.DETECTION

    #: Classes this module reports. Empty means "report everything the model gives".
    default_labels: ClassVar[Tuple[str, ...]] = ()
    #: Weights used when the config does not name any.
    default_model: ClassVar[str] = DEFAULT_COCO_MODEL
    #: True for detectors whose classes are absent from any stock model, so a
    #: user-supplied model path is mandatory.
    requires_own_model: ClassVar[bool] = False
    #: Human-readable hint used when a required model is missing.
    model_hint: ClassVar[str] = ""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._backend: DetectorBackend | None = None

        configured = self.params.get("labels")
        self._labels: Tuple[str, ...] = (
            tuple(str(label) for label in configured)
            if configured
            else tuple(self.default_labels)
        )
        self._exclude: frozenset[str] = frozenset(
            str(label) for label in (self.params.get("exclude_labels") or ())
        )
        self._dedupe_iou = float(self.params.get("dedupe_iou", 0.0) or 0.0)
        self._min_confidence = float(self.params.get("confidence_threshold", 0.40))
        self._min_box_area = float(self.params.get("min_box_area", 0.0))

        candidate = self.params.get("candidate_threshold")
        self._candidate_confidence: float | None = (
            float(candidate) if candidate is not None else None
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        model = self.params.get("model")

        if self._candidate_confidence is not None and not (
            0.0 <= self._candidate_confidence < self._min_confidence
        ):
            raise ModuleInitializationError(
                f"{self.module_name}.candidate_threshold "
                f"({self._candidate_confidence}) must be >= 0 and below "
                f"confidence_threshold ({self._min_confidence})"
            )

        if self.requires_own_model and not model:
            raise ModuleInitializationError(self._missing_model_message())

        backend_params = dict(self.params)
        backend_params["model"] = model or self.default_model

        backend_name = str(self.params.get("backend", DEFAULT_BACKEND))
        try:
            self._backend = create_detector(backend_name, backend_params)
            self._backend.load()
        except BackendError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        self.log.info(
            "Ready: backend=%s, labels=%s, confidence>=%.2f",
            self._backend.description,
            list(self._labels) or "(all)",
            self._min_confidence,
        )

    def cleanup(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        super().cleanup()

    def _missing_model_message(self) -> str:
        return (
            f"Module '{self.module_name}' needs a model that can detect "
            f"{list(self._labels)}, and no stock COCO model contains those "
            f"classes.\n"
            f"Set modules.{self.module_name}.model to a path for a "
            f"custom-trained detector, for example:\n"
            f"    {self.module_name}:\n"
            f"      enabled: true\n"
            f"      model: computer_vision/Model/custom/{self.module_name}.pt\n"
            + (f"{self.model_hint}\n" if self.model_hint else "")
        )

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        if self._backend is None:
            raise RuntimeError(
                f"{type(self).__name__}.process() called before initialize()"
            )

        raw = self._backend.infer(context.frame)
        detections = self._filter(raw, context)
        context.add_detections(self.module_name, detections)

        if self._candidate_confidence is not None:
            context.add_candidates(
                self.module_name,
                self._filter(raw, context, self._candidate_confidence, self._min_confidence),
            )

        return self.make_result(
            context,
            detections=[det.to_dict() for det in detections],
            data=self.summarise(detections),
        )

    def summarise(self, detections: Sequence[Detection]) -> Dict[str, Any]:
        """Module-level summary placed in the result's ``data``.

        Overridden by subclasses that report more than a count.
        """
        counts: Dict[str, int] = {}
        for detection in detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        return {"count": len(detections), "by_class": counts}

    def _filter(
        self,
        raw_detections: Iterable[RawDetection],
        context: FrameContext,
        min_confidence: float | None = None,
        below: float | None = None,
    ) -> List[Detection]:
        """Apply this module's class, confidence and size rules.

        With the defaults this is the module's reported output. Passing
        ``min_confidence``/``below`` selects a different score band with the
        same class and size rules - used for tracker candidates.
        """
        floor = self._min_confidence if min_confidence is None else min_confidence
        detections: List[Detection] = []

        for raw in raw_detections:
            if self._labels and raw.label not in self._labels:
                continue
            if raw.label in self._exclude:
                continue
            if raw.confidence < floor:
                continue
            if below is not None and raw.confidence >= below:
                continue

            bbox = BBox.from_xyxy(raw.xyxy).clip(context.width, context.height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            if self._min_box_area and bbox.area < self._min_box_area:
                continue

            detections.append(
                Detection(
                    label=raw.label,
                    confidence=raw.confidence,
                    bbox=bbox,
                    class_id=raw.class_id,
                    source_module=self.module_name,
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        if self._dedupe_iou > 0:
            detections = self._dedupe(detections)
        return detections

    def _dedupe(self, detections: List[Detection]) -> List[Detection]:
        """Keep the most confident of differently-labelled boxes on one spot.

        Same-label overlaps are left alone: the model's NMS already decided
        those are two objects (two people standing close).
        """
        kept: List[Detection] = []
        for detection in detections:  # most confident first
            if any(
                other.label != detection.label
                and other.bbox.iou(detection.bbox) >= self._dedupe_iou
                for other in kept
            ):
                continue
            kept.append(detection)
        return kept
