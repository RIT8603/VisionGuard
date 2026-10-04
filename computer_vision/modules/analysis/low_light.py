"""Low-Light module.

Two responsibilities that belong together: notice when the scene is dark, and
(optionally) brighten the frame so the detectors that run after it can see.

**Noticing.** The mean brightness of the frame is tracked, and
``low_light_started`` / ``low_light_ended`` are raised when it crosses the
threshold and stays there. Two thresholds (enter and leave) and a frame count
keep a passing shadow or a headlight from flipping the state.

**Brightening** (``enhance: true``). While the scene is dark, the frame the
rest of the pipeline sees is replaced by a brightened copy: a gamma curve
that lifts the mean towards a target, then local contrast equalisation
(CLAHE) on the lightness channel. Colour is left alone.

For the brightening to help anyone, this module has to run **before** the
detectors - list it first under ``modules:`` in config.yaml. It warns once if
it finds detections already published for the frame.

This recovers detail that is in the image but too dark for the model. It
cannot recover what the sensor never captured: a truly black frame stays
empty, and heavy sensor noise is amplified along with the signal. Infrared
cameras solve night better than software does.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.analysis.common import make_event, parse_severity, positive_number
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

EVENT_STARTED = "low_light_started"
EVENT_ENDED = "low_light_ended"


def mean_brightness(frame: np.ndarray) -> float:
    """Mean luma (0-255) of a BGR frame, measured on a small copy."""
    height, width = frame.shape[:2]
    if width > 160:
        frame = cv2.resize(frame, (160, max(1, height * 160 // width)), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def enhance_frame(frame: np.ndarray, target: float = 110.0, clip_limit: float = 2.0) -> np.ndarray:
    """Brighten a dark BGR frame: gamma towards ``target``, then CLAHE on L."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    mean = max(float(lightness.mean()), 1.0)
    if mean < target:
        gamma = float(np.clip(np.log(target / 255.0) / np.log(mean / 255.0), 0.3, 1.0))
        table = (np.linspace(0.0, 1.0, 256) ** gamma * 255.0).astype(np.uint8)
        lightness = cv2.LUT(lightness, table)

    if clip_limit > 0:
        lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lightness)

    lab[:, :, 0] = lightness
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class LowLightModule(BaseAIModule):
    """Reports darkness and optionally brightens the frame for later modules.

    Config block::

        low_light:
          enabled: true
          threshold: 50          # mean brightness (0-255) below this = dark
          recover: 65            # ...and back above this = light again
          min_frames: 15         # frames a change must hold before it counts
          enhance: true          # brighten the frame while dark
          target_brightness: 110
          clahe_clip: 2.0        # 0 turns local contrast equalisation off
          severity: info
    """

    module_name = "low_light"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ()

    def initialize(self) -> None:
        self._threshold = positive_number(self.params, "threshold", 50.0)
        self._recover = positive_number(self.params, "recover", self._threshold + 15.0)
        if self._recover < self._threshold:
            raise ModuleInitializationError(
                "low_light.recover must be >= threshold (it is the brightness "
                "at which the scene counts as light again)"
            )
        self._min_frames = max(1, int(self.params.get("min_frames", 15)))
        self._enhance = bool(self.params.get("enhance", False))
        self._target = positive_number(self.params, "target_brightness", 110.0)
        self._clip = positive_number(self.params, "clahe_clip", 2.0, allow_zero=True)
        self._severity = parse_severity(self.params, Severity.INFO)

        self._dark = False
        self._streak = 0
        self._warned_order = False
        self._total = 0
        self.log.info(
            "Ready: dark below %.0f, light above %.0f, enhance=%s",
            self._threshold, self._recover, "on" if self._enhance else "off",
        )

    @property
    def is_dark(self) -> bool:
        return self._dark

    def process(self, context: FrameContext) -> ModuleResult:
        brightness = mean_brightness(context.frame)
        events = []

        # Hysteresis: while light, only "below threshold" argues for a flip;
        # while dark, only "above recover" does.
        wants_flip = brightness >= self._recover if self._dark else brightness < self._threshold
        self._streak = self._streak + 1 if wants_flip else 0
        if self._streak >= self._min_frames:
            self._dark = not self._dark
            self._streak = 0
            events.append(
                make_event(
                    self.module_name,
                    EVENT_STARTED if self._dark else EVENT_ENDED,
                    context,
                    self._severity,
                    data={"brightness": round(brightness, 1), "threshold": self._threshold},
                )
            )

        enhanced = False
        if self._dark and self._enhance:
            if context.detections and not self._warned_order:
                self._warned_order = True
                self.log.warning(
                    "Detectors already ran on this frame, so brightening it helps "
                    "nothing. List low_light FIRST under modules: in config.yaml."
                )
            context.frame = enhance_frame(context.frame, self._target, self._clip)
            enhanced = True

        for event in events:
            context.add_event(event)
            self.log.info("%s (brightness %.0f)", event.event_type, brightness)
        self._total += len(events)

        data: Dict[str, Any] = {
            "brightness": round(brightness, 1),
            "low_light": self._dark,
            "enhanced": enhanced,
            "event_total": self._total,
        }
        if events:
            data["events"] = [event.to_dict() for event in events]
        return self.make_result(context, data=data)
