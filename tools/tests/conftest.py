"""Shared pytest fixtures.

No test in this suite requires a camera, a display or a model. Phase 1 must be
verifiable on a headless CI machine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from computer_vision.core.config import VisionGuardConfig  # noqa: E402
from computer_vision.core.context import FrameContext  # noqa: E402
from computer_vision.inference import factory as detector_factory  # noqa: E402
from computer_vision.inference.base import DetectorBackend, ModelLoadError, RawDetection  # noqa: E402
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType  # noqa: E402
from computer_vision.modules.registry import ModuleRegistry  # noqa: E402

_ENV_VARS_TO_CLEAR = (
    "VIDEO_SOURCE",
    "VIDEO_SOURCE_ID",
    "VIDEO_BACKEND",
    "VIDEO_DEVICE_ID",
    "VIDEO_WIDTH",
    "VIDEO_HEIGHT",
    "VIDEO_FPS",
    "VIDEO_PATH",
    "VIDEO_LOOP",
    "RTSP_URL",
    "RTSP_TRANSPORT",
    "LOG_LEVEL",
    "LOG_FILE",
    "SHOW_WINDOW",
    "TARGET_WIDTH",
    "MIRROR",
    "MAX_FPS",
    "CONFIG",
)


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry and detector cache are process-global; isolate each test."""
    ModuleRegistry.reset()
    detector_factory.reset_detector_cache()
    yield
    ModuleRegistry.reset()
    detector_factory.reset_detector_cache()


def inner_backend(module):
    """The concrete backend behind a module, unwrapping model sharing.

    Tests that inspect the fake backend's own counters need the real object,
    not the :class:`SharedDetector` the factory hands out by default.
    """
    backend = module._backend
    return getattr(backend, "inner", backend)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Stop a developer's real .env / shell from leaking into assertions."""
    for name in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"VISIONGUARD_{name}", raising=False)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_context(blank_frame) -> FrameContext:
    return FrameContext(
        frame=blank_frame,
        frame_id=1,
        source_id="test-source",
        monotonic_ts=0.0,
    )


@pytest.fixture
def minimal_config() -> VisionGuardConfig:
    return VisionGuardConfig.model_validate(
        {
            "video": {"source": "webcam", "device_id": 0},
            "processing": {"show_window": False, "target_width": None},
            "modules": {},
        }
    )


class FakeDetectorBackend(DetectorBackend):
    """A detector that returns whatever it was told to, with no model.

    Registered as backend ``"fake"`` for every test, so detection-dependent
    tests never need torch, weights or a network.
    """

    name = "fake"
    #: Returned on every call.
    scripted: List[RawDetection] = []
    #: One entry per call, for tests that need detections to move between
    #: frames. Takes precedence over ``scripted``; the last entry repeats.
    scripted_sequence: List[List[RawDetection]] | None = None
    fail_on_load = False

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.load_calls = 0
        self.unload_calls = 0
        self.frames_seen = 0

    def load(self) -> None:
        self.load_calls += 1
        if FakeDetectorBackend.fail_on_load:
            raise ModelLoadError("no weights here")
        self._loaded = True

    def infer(self, frame) -> List[RawDetection]:
        sequence = FakeDetectorBackend.scripted_sequence
        if sequence:
            index = min(self.frames_seen, len(sequence) - 1)
            self.frames_seen += 1
            return list(sequence[index])

        self.frames_seen += 1
        return list(FakeDetectorBackend.scripted)

    def unload(self) -> None:
        self.unload_calls += 1
        super().unload()


@pytest.fixture(autouse=True)
def fake_detector_backend(monkeypatch):
    """Make backend 'fake' available, and reset its script between tests."""
    FakeDetectorBackend.scripted = []
    FakeDetectorBackend.scripted_sequence = None
    FakeDetectorBackend.fail_on_load = False
    monkeypatch.setitem(
        detector_factory.BACKENDS, "fake", "tests.conftest:FakeDetectorBackend"
    )
    yield FakeDetectorBackend
    FakeDetectorBackend.scripted = []
    FakeDetectorBackend.scripted_sequence = None
    FakeDetectorBackend.fail_on_load = False


def make_module(
    name: str,
    *,
    kind: ModuleType = ModuleType.DETECTION,
    deps: Tuple[str, ...] = (),
    fail_on_process: bool = False,
    fail_on_init: bool = False,
) -> Type[BaseAIModule]:
    """Build a throwaway AI module class for registry and pipeline tests."""

    class _TestModule(BaseAIModule):
        module_name = name
        module_type = kind
        version = "0.0.1"
        requires = deps

        def __init__(self, params: Dict[str, Any] | None = None) -> None:
            super().__init__(params)
            self.initialized = False
            self.cleaned_up = False
            self.processed_frames: List[int] = []

        def initialize(self) -> None:
            if fail_on_init:
                raise RuntimeError(f"{name} cannot load its model")
            self.initialized = True

        def process(self, context: FrameContext) -> ModuleResult:
            if fail_on_process:
                raise RuntimeError(f"{name} blew up")
            self.processed_frames.append(context.frame_id)
            return self.make_result(context, data={"ran": True})

        def cleanup(self) -> None:
            super().cleanup()
            self.cleaned_up = True

    _TestModule.__name__ = name.title().replace("_", "") + "TestModule"
    return _TestModule
