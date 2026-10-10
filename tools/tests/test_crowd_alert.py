"""crowd_alert: crowd_detection's count turned into crowd events.

Tested on hand-built crowd_detection results - the module's logic is time,
not inference, so no model is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from computer_vision.core.config import load_config
from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.analysis.crowd_alert import CrowdAlertModule
from computer_vision.modules.base import ModuleInitializationError, ModuleResult, ModuleType
from computer_vision.modules.registry import ModuleRegistry


def frame(t: float, count: int, threshold: int = 5) -> FrameContext:
    context = FrameContext(
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=int(t * 10),
        source_id="cam-a",
        monotonic_ts=t,
    )
    context.set_result(
        "crowd_detection",
        ModuleResult(
            module="crowd_detection",
            module_type=ModuleType.ANALYSIS,
            version="1.0.0",
            frame_id=context.frame_id,
            data={"count": count, "threshold": threshold, "is_crowded": count >= threshold},
        ),
    )
    return context


def make(**params) -> CrowdAlertModule:
    module = CrowdAlertModule(params)
    module.initialize()
    return module


def run(module: CrowdAlertModule, t: float, count: int):
    context = frame(t, count)
    module.process(context)
    return [event.event_type for event in context.events], context


class TestCrowdAlert:
    def test_brief_crowd_is_not_reported(self):
        module = make(min_seconds=3)
        assert run(module, 0.0, 6)[0] == []
        assert run(module, 2.0, 6)[0] == []
        assert run(module, 2.5, 2)[0] == []
        # The clock restarted when the crowd broke up.
        assert run(module, 4.0, 6)[0] == []

    def test_lasting_crowd_is_reported_once(self):
        module = make(min_seconds=3, cooldown_seconds=0)
        run(module, 0.0, 6)
        events, context = run(module, 3.0, 7)
        assert events == ["crowd_formed"]
        event = context.events[0]
        assert event.severity is Severity.WARNING
        assert event.data["count"] == 7 and event.data["threshold"] == 5
        assert run(module, 4.0, 7)[0] == []
        assert run(module, 100.0, 7)[0] == []

    def test_cooldown_repeats_while_crowd_lasts(self):
        module = make(min_seconds=0, cooldown_seconds=10)
        assert run(module, 0.0, 6)[0] == ["crowd_formed"]
        assert run(module, 9.0, 6)[0] == []
        assert run(module, 10.0, 6)[0] == ["crowd_formed"]

    def test_clears_after_clear_seconds_with_peak(self):
        module = make(min_seconds=0, clear_seconds=5)
        run(module, 0.0, 6)
        run(module, 1.0, 9)
        assert run(module, 2.0, 1)[0] == []
        # Back above the threshold briefly: the calm clock restarts.
        run(module, 3.0, 6)
        assert run(module, 4.0, 1)[0] == []
        events, context = run(module, 9.0, 0)
        assert events == ["crowd_cleared"]
        assert context.events[0].severity is Severity.INFO
        assert context.events[0].data["peak_count"] == 9
        # A new crowd is a new crowd_formed.
        assert run(module, 10.0, 6)[0] == ["crowd_formed"]

    def test_report_cleared_can_be_turned_off(self):
        module = make(min_seconds=0, clear_seconds=0, report_cleared=False)
        run(module, 0.0, 6)
        assert run(module, 1.0, 0)[0] == []
        assert run(module, 2.0, 6)[0] == ["crowd_formed"]

    def test_missing_crowd_detection_result_is_quiet(self):
        module = make(min_seconds=0)
        context = FrameContext(
            frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_id=0, source_id="c", monotonic_ts=0.0
        )
        result = module.process(context)
        assert context.events == [] and result.data["crowd_active"] is False

    @pytest.mark.parametrize("params", [{"min_seconds": -1}, {"severity": "loud"}])
    def test_bad_config_fails_readably(self, params):
        with pytest.raises(ModuleInitializationError):
            make(**params)


class TestCrowdAlertWiring:
    def test_registered_and_requires_crowd_detection(self):
        assert ModuleRegistry.resolve("crowd_alert") is CrowdAlertModule
        assert CrowdAlertModule.requires == ("crowd_detection",)

    def test_config_has_it_disabled_by_default(self):
        config = load_config("configs/config.yaml", use_env=False)
        assert "crowd_alert" in config.modules
        assert config.modules["crowd_alert"].enabled is False
