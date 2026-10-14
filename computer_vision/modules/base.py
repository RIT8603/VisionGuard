"""The common AI module interface.

Every AI capability in VisionGuard — detection, recognition, tracking, later
analysis — is a subclass of :class:`BaseAIModule`. The pipeline knows only this
interface, which is what keeps module-specific logic out of the core.

A module declares its dependencies with :attr:`requires`, naming the modules
whose results it consumes. The registry uses that to order execution, so
``person_tracking`` runs after ``person_detection`` without the pipeline
hard-coding either name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, List, Tuple

from computer_vision.core.context import FrameContext
from computer_vision.utils.logger import get_logger
from computer_vision.utils.timing import Stopwatch

logger = get_logger(__name__)


class ModuleType(str, Enum):
    """Category of work a module performs."""

    DETECTION = "detection"
    RECOGNITION = "recognition"
    TRACKING = "tracking"
    ANALYSIS = "analysis"


class ModuleStatus(str, Enum):
    """Lifecycle state of a module instance."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class ModuleError(Exception):
    """Base class for module failures."""


class ModuleInitializationError(ModuleError):
    """A module could not load its model or otherwise failed to initialize."""


@dataclass
class ModuleResult:
    """The standardised output every module returns.

    ``to_dict`` produces the schema the dashboard and any future consumer will
    read, matching the shape defined in PROJECT_SPEC.md:

        {
            "module": "person_detection",
            "timestamp": "2026-09-24T09:15:03.412+00:00",
            "detections": [
                {"class": "person", "confidence": 0.94, "bbox": [x1, y1, x2, y2]}
            ]
        }
    """

    module: str
    module_type: ModuleType
    version: str
    frame_id: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float = 0.0
    detections: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "module": self.module,
            "module_type": self.module_type.value,
            "version": self.version,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.detections:
            payload["detections"] = self.detections
        if self.tracks:
            payload["tracks"] = self.tracks
        if self.data:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        return payload


class BaseAIModule(ABC):
    """Contract for every AI module.

    Subclasses must declare the four class attributes and implement
    :meth:`initialize` and :meth:`process`.

    Example skeleton for a future module::

        @register_module
        class FireDetectionModule(BaseAIModule):
            module_name = "fire_detection"
            module_type = ModuleType.DETECTION
            version = "1.0.0"
            requires = ()

            def initialize(self) -> None:
                self._model = load_model(self.params.get("model_path"))

            def process(self, context: FrameContext) -> ModuleResult:
                ...

    Rules enforced by review, not by code:
        * one module, one responsibility
        * no notifications, database writes or business rules inside a module
        * never reach into another module instance; communicate via FrameContext
    """

    #: Unique identifier. Must match the key used in ``config.yaml``.
    module_name: ClassVar[str] = ""
    #: Category used for grouping and for sanity checks.
    module_type: ClassVar[ModuleType] = ModuleType.DETECTION
    #: Semantic version of the module implementation.
    version: ClassVar[str] = "0.1.0"
    #: Names of modules whose results this module consumes.
    requires: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        """Modules this one needs, given its configuration.

        Defaults to the static :attr:`requires`. Higher-level modules override
        it when what they consume is a configuration choice - a restricted
        zone might watch ``person_tracking``, ``vehicle_tracking`` or both, and
        the registry must order and validate whichever was chosen.

        Called by the registry *before* instantiation, so it must not depend on
        instance state.
        """
        return cls.requires

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        if not self.module_name:
            raise ModuleError(
                f"{type(self).__name__} must define a non-empty 'module_name'"
            )
        self.params: Dict[str, Any] = dict(params or {})
        self.status: ModuleStatus = ModuleStatus.UNINITIALIZED
        self._last_result: ModuleResult | None = None
        self._consecutive_failures = 0
        self.log = get_logger(f"module.{self.module_name}")

    # ------------------------------------------------------------------ #
    # Lifecycle — subclasses implement
    # ------------------------------------------------------------------ #

    @abstractmethod
    def initialize(self) -> None:
        """Load models and allocate resources.

        Called once, by the registry, and only for modules enabled in the
        configuration. A disabled module never reaches this method, which is
        how "disabled modules must not load their models" is guaranteed.

        Raises:
            ModuleInitializationError: when the module cannot become ready.
        """

    @abstractmethod
    def process(self, context: FrameContext) -> ModuleResult:
        """Analyse one frame and publish results into ``context``.

        Implementations should write their output to the context
        (``context.add_detections(self.module_name, ...)``) *and* return a
        :class:`ModuleResult` describing the same data.
        """

    def reset(self) -> None:
        """Reset internal state (e.g. on camera switch or loop rewind)."""
        pass

    def cleanup(self) -> None:
        """Release resources. Safe to call more than once.

        A module that was disabled for repeated failures keeps its ``FAILED``
        status through shutdown, so a post-mortem still shows what broke.
        """
        if self.status is not ModuleStatus.FAILED:
            self.status = ModuleStatus.STOPPED

    # ------------------------------------------------------------------ #
    # Pipeline-facing helpers
    # ------------------------------------------------------------------ #

    def get_result(self) -> ModuleResult | None:
        """The most recent result produced by this module."""
        return self._last_result

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def safe_process(self, context: FrameContext) -> ModuleResult:
        """Run :meth:`process` with timing and error containment.

        A module that raises does not bring down the pipeline: the exception is
        captured into the returned :class:`ModuleResult` and recorded on the
        context. The pipeline decides what to do about repeated failures.
        """
        with Stopwatch() as watch:
            try:
                result = self.process(context)
                self._consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - containment is the point
                self._consecutive_failures += 1
                self.log.exception("process() failed on frame %s", context.frame_id)
                context.record_error(self.module_name, f"{type(exc).__name__}: {exc}")
                result = self.empty_result(context.frame_id, error=f"{type(exc).__name__}: {exc}")

        result.latency_ms = watch.elapsed_ms
        self._last_result = result
        context.set_result(self.module_name, result)
        return result

    def empty_result(self, frame_id: int, error: str | None = None) -> ModuleResult:
        """A well-formed result carrying no detections."""
        return ModuleResult(
            module=self.module_name,
            module_type=self.module_type,
            version=self.version,
            frame_id=frame_id,
            error=error,
        )

    def make_result(self, context: FrameContext, **kwargs: Any) -> ModuleResult:
        """Build a result pre-filled with this module's identity."""
        return ModuleResult(
            module=self.module_name,
            module_type=self.module_type,
            version=self.version,
            frame_id=context.frame_id,
            timestamp=context.timestamp_iso,
            **kwargs,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_name,
            "type": self.module_type.value,
            "version": self.version,
            "status": self.status.value,
            "requires": list(self.requires),
            "params": self.params,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.module_name} v{self.version} {self.status.value}>"
