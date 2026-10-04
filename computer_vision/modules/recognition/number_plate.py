"""Number Plate Recognition (ANPR) module.

One responsibility: read the registration plate of vehicles that
``vehicle_tracking`` is already following.

For each tracked vehicle it crops the vehicle from the frame, finds the plate
inside that crop, and - when the plate is large enough to be legible - reads
its text. A plate is reported once per vehicle, after the same text has been
read on several frames, so one misread frame does not become a record.

Two details matter more than the networks do:

* **It reads from the original capture, not the resized frame.** The pipeline
  works at ``target_width`` (960 px), where a plate is a few dozen pixels
  wide. The crop is taken from ``FrameContext.original_frame`` instead, and
  the result is mapped back to frame coordinates for drawing.
* **It does not guess.** A plate narrower than ``min_plate_width`` pixels is
  drawn as a box labelled ``plate`` and not read at all: below that size the
  characters are not in the image, and the OCR network will happily produce a
  confident-looking wrong answer.

A registration number identifies a vehicle and, through it, usually a person.
Treat what this module produces as personal data. It is off by default.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from computer_vision.core.context import BBox, Detection, FrameContext, Severity
from computer_vision.inference.base import BackendError
from computer_vision.modules.analysis.common import (
    SubjectMemory,
    make_event,
    parse_severity,
    positive_number,
)
from computer_vision.modules.analysis.geometry import subject_key
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_DETECTOR = "computer_vision/Model/anpr/plate_detector_yolov9t_384.onnx"
DEFAULT_OCR = "computer_vision/Model/anpr/plate_ocr_cct_s_v2_global.onnx"
EVENT_PLATE = "plate_read"
UNREAD_LABEL = "plate"


@dataclass
class _Vehicle:
    last_checked_frame: int = -(10**9)
    votes: Counter = field(default_factory=Counter)
    confidences: Dict[str, float] = field(default_factory=dict)
    reported: str = ""
    #: Latest plate as fractions of the vehicle box, so it can be redrawn on
    #: frames where the detector is not run.
    relative_box: Tuple[float, float, float, float] | None = None
    plate_confidence: float = 0.0
    plate_width: int = 0
    region: str = ""

    @property
    def best(self) -> Tuple[str, int]:
        return self.votes.most_common(1)[0] if self.votes else ("", 0)


class NumberPlateRecognitionModule(BaseAIModule):
    """Finds and reads number plates on tracked vehicles.

    Config block::

        number_plate_recognition:
          enabled: true
          source_module: vehicle_tracking
          detector_model: computer_vision/Model/anpr/plate_detector_yolov9t_384.onnx
          ocr_model: computer_vision/Model/anpr/plate_ocr_cct_s_v2_global.onnx
          device: cpu              # cpu | gpu (OpenVINO)
          plate_confidence: 0.4    # plate detector score
          min_plate_width: 80      # px in the original capture; smaller is not read
          read_confidence: 0.85    # mean character probability to accept a read
          min_characters: 4
          votes: 2                 # identical reads before a plate is reported
          interval_frames: 5       # look at each vehicle every N frames
          max_vehicles: 2          # vehicles examined per frame, largest first
          severity: info

    Requires ``vehicle_tracking`` and the ``openvino`` package.
    """

    module_name = "number_plate_recognition"
    module_type = ModuleType.RECOGNITION
    version = "1.0.0"
    requires = ("vehicle_tracking",)

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        return (str((params or {}).get("source_module") or "vehicle_tracking"),)

    def initialize(self) -> None:
        self._source = self.required_modules(self.params)[0]
        self._plate_confidence = positive_number(self.params, "plate_confidence", 0.4)
        self._min_width = positive_number(self.params, "min_plate_width", 80.0, allow_zero=True)
        self._read_confidence = positive_number(self.params, "read_confidence", 0.85)
        self._min_characters = max(1, int(self.params.get("min_characters", 4)))
        self._votes = max(1, int(self.params.get("votes", 2)))
        self._interval = max(1, int(self.params.get("interval_frames", 5)))
        self._max_vehicles = max(1, int(self.params.get("max_vehicles", 2)))
        # A plate is never more than about half a vehicle's width, so a
        # vehicle narrower than twice the minimum plate cannot carry a
        # readable one. Skipping it saves the plate detector's time, which is
        # the expensive part of this module.
        self._min_vehicle_width = 2.0 * self._min_width
        self._severity = parse_severity(self.params, Severity.INFO)

        try:
            self._reader = self._build_reader()
        except BackendError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        self._memory: SubjectMemory[_Vehicle] = SubjectMemory(forget_seconds=5.0)
        self._total = 0
        self.log.info(
            "Ready: reading plates >= %.0fpx wide on '%s' vehicles (confidence >= %.2f, %d votes)",
            self._min_width, self._source, self._read_confidence, self._votes,
        )

    def _build_reader(self):
        """Load the two networks. A seam so tests need neither file."""
        from computer_vision.inference.plate_reader import PlateReader

        return PlateReader(
            str(self.params.get("detector_model") or DEFAULT_DETECTOR),
            str(self.params.get("ocr_model") or DEFAULT_OCR),
            detector_sha256=self.params.get("detector_sha256"),
            ocr_sha256=self.params.get("ocr_sha256"),
            device=str(self.params.get("device", "cpu")),
        )

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        image = context.original_frame if context.original_frame is not None else context.frame
        scale_x = image.shape[1] / context.width
        scale_y = image.shape[0] / context.height

        tracks = sorted(
            context.get_tracks(self._source), key=lambda t: t.bbox.area, reverse=True
        )
        examined = 0
        events = []
        detections: List[Detection] = []

        for track in tracks:
            key = subject_key(track.source_module, track.track_id)
            state = self._memory.get(key) or _Vehicle()
            self._memory.put(key, state, now)

            due = context.frame_id - state.last_checked_frame >= self._interval
            big_enough = track.bbox.width * scale_x >= self._min_vehicle_width
            if not big_enough:
                state.relative_box = None
            if due and big_enough and examined < self._max_vehicles:
                examined += 1
                state.last_checked_frame = context.frame_id
                self._examine(track, state, image, scale_x, scale_y)

                text, count = state.best
                if text and count >= self._votes and state.reported != text:
                    state.reported = text
                    events.append(
                        make_event(
                            self.module_name, EVENT_PLATE, context, self._severity,
                            subject={"track_id": track.track_id, "class": track.label},
                            data={
                                "plate": text,
                                "region": state.region,
                                "read_confidence": round(state.confidences.get(text, 0.0), 3),
                                "plate_width": state.plate_width,
                                "votes": count,
                            },
                        )
                    )

            detection = self._as_detection(track, state)
            if detection is not None:
                detections.append(detection)

        self._memory.expire(now)
        context.add_detections(self.module_name, detections)
        for event in events:
            context.add_event(event)
            self.log.info("%s: %s", event.describe(), event.data["plate"])
        self._total += len(events)

        return self.make_result(
            context,
            detections=[det.to_dict() for det in detections],
            data={
                "plates": [
                    {"track_id": det.extra["track_id"], "text": det.extra.get("text", ""),
                     "plate_width": det.extra["plate_width"]}
                    for det in detections
                ],
                "examined": examined,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def _examine(self, track, state: _Vehicle, image: np.ndarray, scale_x: float, scale_y: float) -> None:
        """Look for a plate on one vehicle and, if legible, read it."""
        x1 = int(max(0, track.bbox.x1 * scale_x))
        y1 = int(max(0, track.bbox.y1 * scale_y))
        x2 = int(min(image.shape[1], track.bbox.x2 * scale_x))
        y2 = int(min(image.shape[0], track.bbox.y2 * scale_y))
        if x2 - x1 < 16 or y2 - y1 < 16:
            state.relative_box = None
            return

        crop = image[y1:y2, x1:x2]
        plates = self._reader.detect(crop, self._plate_confidence)
        if not plates:
            state.relative_box = None
            return

        plate = plates[0]
        state.relative_box = (
            plate.x1 / crop.shape[1], plate.y1 / crop.shape[0],
            plate.x2 / crop.shape[1], plate.y2 / crop.shape[0],
        )
        state.plate_confidence = plate.confidence
        state.plate_width = plate.width

        if plate.width < self._min_width:
            return  # visible, but too small to read honestly

        read = self._reader.read(crop[plate.y1:plate.y2, plate.x1:plate.x2])
        if read.confidence < self._read_confidence or len(read.text) < self._min_characters:
            return
        state.votes[read.text] += 1
        state.confidences[read.text] = max(state.confidences.get(read.text, 0.0), read.confidence)
        state.region = read.region

    def _as_detection(self, track, state: _Vehicle) -> Detection | None:
        """The plate box in frame coordinates, labelled with its text if known."""
        if state.relative_box is None:
            return None
        rx1, ry1, rx2, ry2 = state.relative_box
        box = track.bbox
        bbox = BBox(
            box.x1 + rx1 * box.width, box.y1 + ry1 * box.height,
            box.x1 + rx2 * box.width, box.y1 + ry2 * box.height,
        )
        text = state.reported
        extra: Dict[str, Any] = {"track_id": track.track_id, "plate_width": state.plate_width}
        if text:
            extra.update(text=text, region=state.region)
        return Detection(
            label=text or UNREAD_LABEL,
            confidence=state.plate_confidence,
            bbox=bbox,
            source_module=self.module_name,
            extra=extra,
        )

    def cleanup(self) -> None:
        if hasattr(self, "_memory"):
            self._memory.clear()
        self._reader = None
        super().cleanup()
