"""Crowd alert module.

One responsibility: turn crowd measurements into crowd **events**.

``crowd_detection`` counts people and says whether the count is at or above
its threshold, every frame. A single crowded frame is not a crowd - a group
walking past for half a second crosses the threshold too - so this module
waits until the crowd has *lasted* ``min_seconds`` before reporting it, and
reports the crowd clearing the same way.

It runs no model and reads no detections: the count and the threshold come
from ``crowd_detection``'s result, so there is exactly one place the
threshold is set.
"""

from __future__ import annotations

from typing import Any, Dict

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.analysis.common import make_event, parse_severity, positive_number
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType

SOURCE_MODULE = "crowd_detection"
EVENT_CROWD_FORMED = "crowd_formed"
EVENT_CROWD_CLEARED = "crowd_cleared"


class CrowdAlertModule(BaseAIModule):
    """Emits ``crowd_formed`` / ``crowd_cleared`` when a crowd persists.

    Config block::

        crowd_alert:
          enabled: true
          min_seconds: 3          # crowded this long before it counts
          clear_seconds: 5        # below threshold this long before it clears
          cooldown_seconds: 60    # repeat crowd_formed while it lasts
          report_cleared: true    # also emit crowd_cleared (severity info)
          severity: warning
    """

    module_name = "crowd_alert"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = (SOURCE_MODULE,)

    def initialize(self) -> None:
        self._min_seconds = positive_number(self.params, "min_seconds", 3.0, allow_zero=True)
        self._clear_seconds = positive_number(self.params, "clear_seconds", 5.0, allow_zero=True)
        self._cooldown = positive_number(self.params, "cooldown_seconds", 60.0, allow_zero=True)
        self._report_cleared = bool(self.params.get("report_cleared", True))
        self._severity = parse_severity(self.params, Severity.WARNING)
        self._reset()
        self._total = 0
        self.log.info(
            "Ready: crowd after %.1fs above crowd_detection's threshold, clears after %.1fs",
            self._min_seconds, self._clear_seconds,
        )

    def _reset(self) -> None:
        self._active = False          # a crowd has been reported and not cleared
        self._crowded_since: float | None = None
        self._calm_since: float | None = None
        self._reported_at: float | None = None
        self._peak = 0

    def process(self, context: FrameContext) -> ModuleResult:
        now = context.monotonic_ts
        result = context.get_result(SOURCE_MODULE)
        data: Dict[str, Any] = getattr(result, "data", None) or {}
        count = int(data.get("count", 0))
        threshold = data.get("threshold")
        crowded = bool(data.get("is_crowded", False))
        events = []

        if crowded:
            self._calm_since = None
            if self._crowded_since is None:
                self._crowded_since = now
            lasted = now - self._crowded_since
            if self._active:
                self._peak = max(self._peak, count)
            due = lasted >= self._min_seconds and (
                not self._active
                or (
                    self._cooldown > 0
                    and self._reported_at is not None
                    and now - self._reported_at >= self._cooldown
                )
            )
            if due:
                if not self._active:
                    self._peak = count
                self._active = True
                self._reported_at = now
                events.append(
                    make_event(
                        self.module_name, EVENT_CROWD_FORMED, context, self._severity,
                        data={"count": count, "threshold": threshold, "seconds": round(lasted, 1)},
                    )
                )
        else:
            self._crowded_since = None
            if self._active:
                if self._calm_since is None:
                    self._calm_since = now
                if now - self._calm_since >= self._clear_seconds:
                    if self._report_cleared:
                        events.append(
                            make_event(
                                self.module_name, EVENT_CROWD_CLEARED, context, Severity.INFO,
                                data={"count": count, "threshold": threshold, "peak_count": self._peak},
                            )
                        )
                    self._reset()

        for event in events:
            context.add_event(event)
            self.log.info("%s (%d people)", event.describe(), count)
        self._total += len(events)

        return self.make_result(
            context,
            data={
                "crowd_active": self._active,
                "count": count,
                "crowded_for": round(now - self._crowded_since, 1) if self._crowded_since is not None else 0.0,
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    def cleanup(self) -> None:
        self._reset()
        super().cleanup()
