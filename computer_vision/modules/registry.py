"""Module registry — discovery, dependency ordering and construction.

Two ways a module becomes known to the platform:

1. **Declared** (preferred for built-ins): a ``name -> "package.module:Class"``
   entry in :data:`BUILTIN_MODULES`. The Python module is imported *only* when
   the module is enabled. This is what guarantees that a disabled module never
   drags in its heavy dependencies — enabling ``face_recognition`` should be the
   only thing that ever imports a face-embedding library.

2. **Registered** (for plugins and tests): the :func:`register_module`
   decorator, which records an already-imported class.

The registry also resolves execution order from each module's ``requires``
declaration, so the pipeline never needs to know that tracking follows
detection.
"""

from __future__ import annotations

import importlib
from typing import Dict, Iterable, List, Mapping, Type

from computer_vision.core.config import ModuleConfig, VisionGuardConfig
from computer_vision.modules.base import (
    BaseAIModule,
    ModuleError,
    ModuleInitializationError,
    ModuleStatus,
)
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)


class ModuleNotRegisteredError(ModuleError):
    """Configuration enables a module the platform does not know about."""


class ModuleDependencyError(ModuleError):
    """A module's declared dependency is missing or forms a cycle."""


# --------------------------------------------------------------------------- #
# Built-in module catalogue
# --------------------------------------------------------------------------- #
#
# Every entry maps a configuration key to "import.path:ClassName".
# Entries are added as each phase delivers its module. Commented lines are the
# planned catalogue and exist to document intent — they are not loaded.
#
BUILTIN_MODULES: Dict[str, str] = {
    # -- Phase 2 --
    "person_detection": "computer_vision.modules.detection.person:PersonDetectionModule",
    # -- Phase 3 --
    "person_tracking": "computer_vision.modules.tracking.person:PersonTrackingModule",
    # -- Phase 6 --
    "object_detection": "computer_vision.modules.detection.object:ObjectDetectionModule",
    "vehicle_detection": "computer_vision.modules.detection.vehicle:VehicleDetectionModule",
    "crowd_detection": "computer_vision.modules.detection.crowd:CrowdDetectionModule",
    # These three need custom-trained weights; no stock COCO model contains
    # their classes. They load fine once `model:` points at one.
    "ppe_detection": "computer_vision.modules.detection.ppe:PPEDetectionModule",
    "fire_detection": "computer_vision.modules.detection.fire:FireDetectionModule",
    "smoke_detection": "computer_vision.modules.detection.smoke:SmokeDetectionModule",
    # -- Phase 7 --
    # Identity processing. All three need model files, and the last two need
    # an identity store. Off unless explicitly enabled (Agent.md §24).
    "face_detection": "computer_vision.modules.recognition.face_detection:FaceDetectionModule",
    "face_liveness": "computer_vision.modules.recognition.face_liveness:FaceLivenessModule",
    "face_recognition": "computer_vision.modules.recognition.face_recognition:FaceRecognitionModule",
    "unknown_person": "computer_vision.modules.recognition.unknown_person:UnknownPersonModule",
    # -- Phase 8 --
    "object_tracking": "computer_vision.modules.tracking.object:ObjectTrackingModule",
    "vehicle_tracking": "computer_vision.modules.tracking.vehicle:VehicleTrackingModule",
    # -- Pose and posture analysis --
    "pose_detection": "computer_vision.modules.detection.pose:PoseDetectionModule",
    "fall_detection": "computer_vision.modules.analysis.fall:FallDetectionModule",
    # -- Phase 9 (higher-level analysis) --
    "restricted_zone": "computer_vision.modules.analysis.zone:RestrictedZoneModule",
    "attendance": "computer_vision.modules.analysis.attendance:AttendanceModule",
    "crowd_alert": "computer_vision.modules.analysis.crowd_alert:CrowdAlertModule",
    # -- Behaviour and scene analysis (no model of their own) --
    "low_light": "computer_vision.modules.analysis.low_light:LowLightModule",
    "loitering": "computer_vision.modules.analysis.loitering:LoiteringModule",
    "intrusion_detection": "computer_vision.modules.analysis.intrusion:IntrusionDetectionModule",
    "tailgating": "computer_vision.modules.analysis.tailgating:TailgatingModule",
    "gate_counter": "computer_vision.modules.analysis.gate:GateCounterModule",
    "abandoned_object": "computer_vision.modules.analysis.abandoned_object:AbandonedObjectModule",
    "inventory_monitoring": "computer_vision.modules.analysis.inventory:InventoryMonitoringModule",
    # -- Weapons: stock COCO classes (knife, bat, scissors); guns need custom weights --
    "weapon_detection": "computer_vision.modules.detection.weapon:WeaponDetectionModule",
    # -- Identification beyond faces. Personal data: off unless enabled. --
    "number_plate_recognition": "computer_vision.modules.recognition.number_plate:NumberPlateRecognitionModule",
    "person_reid": "computer_vision.modules.recognition.person_reid:PersonReIdModule",
}


class ModuleRegistry:
    """Central catalogue of AI modules.

    All state is class-level: there is exactly one registry per process.
    """

    _registered: Dict[str, Type[BaseAIModule]] = {}
    _declared: Dict[str, str] = dict(BUILTIN_MODULES)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    @classmethod
    def register(
        cls, module_cls: Type[BaseAIModule], *, override: bool = False
    ) -> Type[BaseAIModule]:
        """Record an imported module class."""
        name = module_cls.module_name
        if not name:
            raise ModuleError(f"{module_cls.__name__} must define 'module_name'")
        if name in cls._registered and not override:
            existing = cls._registered[name]
            if existing is not module_cls:
                raise ModuleError(
                    f"Module name '{name}' is already registered by "
                    f"{existing.__module__}.{existing.__name__}"
                )
        cls._registered[name] = module_cls
        logger.debug("Registered module '%s' (%s)", name, module_cls.__name__)
        return module_cls

    @classmethod
    def declare(cls, name: str, import_path: str) -> None:
        """Declare a module for lazy import: ``"pkg.mod:ClassName"``."""
        cls._declared[name] = import_path

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registration. Primarily for tests."""
        cls._registered.pop(name, None)

    @classmethod
    def reset(cls) -> None:
        """Restore the registry to its built-in state. Primarily for tests."""
        cls._registered.clear()
        cls._declared = dict(BUILTIN_MODULES)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    @classmethod
    def known_names(cls) -> List[str]:
        """Every module name the platform can resolve."""
        return sorted(set(cls._registered) | set(cls._declared))

    @classmethod
    def is_known(cls, name: str) -> bool:
        return name in cls._registered or name in cls._declared

    @classmethod
    def resolve(cls, name: str) -> Type[BaseAIModule]:
        """Return the class for ``name``, importing it if it was only declared.

        Raises:
            ModuleNotRegisteredError: if the name is unknown or the declared
                import path cannot be loaded.
        """
        if name in cls._registered:
            return cls._registered[name]

        import_path = cls._declared.get(name)
        if import_path is None:
            raise ModuleNotRegisteredError(
                f"Unknown AI module '{name}'.\n"
                f"Known modules: {cls.known_names() or '(none yet)'}\n"
                "Either the name is misspelled in config.yaml, or the module "
                "belongs to a later development phase (see ROADMAP.md)."
            )

        module_path, _, class_name = import_path.partition(":")
        if not class_name:
            raise ModuleNotRegisteredError(
                f"Malformed import path for '{name}': {import_path!r} "
                "(expected 'package.module:ClassName')"
            )

        logger.debug("Lazily importing module '%s' from %s", name, import_path)
        try:
            imported = importlib.import_module(module_path)
            module_cls = getattr(imported, class_name)
        except (ImportError, AttributeError) as exc:
            raise ModuleNotRegisteredError(
                f"Could not load AI module '{name}' from {import_path!r}: {exc}"
            ) from exc

        if not (isinstance(module_cls, type) and issubclass(module_cls, BaseAIModule)):
            raise ModuleNotRegisteredError(
                f"{import_path} is not a BaseAIModule subclass"
            )

        cls._registered[name] = module_cls
        return module_cls

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def build_enabled(cls, config: VisionGuardConfig) -> List[BaseAIModule]:
        """Instantiate and initialize every enabled module, in dependency order.

        Disabled modules are never resolved, imported or constructed.

        Raises:
            ModuleNotRegisteredError: an enabled module is unknown.
            ModuleDependencyError: a dependency is not enabled, or there is a cycle.
            ModuleInitializationError: a module failed to initialize.
        """
        enabled = config.enabled_module_names()
        disabled = [n for n in config.modules if n not in enabled]

        logger.info(
            "Modules - enabled: %s | disabled: %s",
            enabled or "(none)",
            disabled or "(none)",
        )
        if not enabled:
            return []

        classes: Dict[str, Type[BaseAIModule]] = {
            name: cls.resolve(name) for name in enabled
        }
        # Dependencies can depend on configuration, so they are resolved once
        # here and reused for validation, ordering and error messages.
        requirements: Dict[str, Tuple[str, ...]] = {
            name: tuple(module_cls.required_modules(config.module_config(name).params))
            for name, module_cls in classes.items()
        }
        cls._validate_dependencies(requirements, classes)
        # Soft ordering: "after X, if X is enabled" shapes the order only.
        ordering: Dict[str, Tuple[str, ...]] = {
            name: requirements[name]
            + tuple(dep for dep in module_cls.run_after if dep in classes and dep != name)
            for name, module_cls in classes.items()
        }
        ordered = cls._topological_order(ordering, enabled)

        logger.info("Module execution order: %s", " -> ".join(ordered))

        instances: List[BaseAIModule] = []
        try:
            for name in ordered:
                instances.append(cls._build_one(name, classes[name], config.module_config(name)))
        except Exception:
            # Never leave half-initialized modules holding models or devices.
            for built in reversed(instances):
                try:
                    built.cleanup()
                except Exception:  # noqa: BLE001 - cleanup must not mask the cause
                    logger.warning("cleanup() failed for '%s'", built.module_name)
            raise

        return instances

    @classmethod
    def _build_one(
        cls, name: str, module_cls: Type[BaseAIModule], module_cfg: ModuleConfig
    ) -> BaseAIModule:
        instance = module_cls(params=module_cfg.params)
        try:
            instance.initialize()
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            instance.status = ModuleStatus.FAILED
            raise ModuleInitializationError(
                f"Module '{name}' failed to initialize: {type(exc).__name__}: {exc}"
            ) from exc
        instance.status = ModuleStatus.READY
        logger.info("Initialized module '%s' v%s", name, instance.version)
        return instance

    # ------------------------------------------------------------------ #
    # Dependency resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_dependencies(
        requirements: Mapping[str, Tuple[str, ...]],
        classes: Mapping[str, Type[BaseAIModule]],
    ) -> None:
        for name, needed in requirements.items():
            missing = [dep for dep in needed if dep not in classes]
            if missing:
                raise ModuleDependencyError(
                    f"Module '{name}' requires {missing}, which "
                    f"{'is' if len(missing) == 1 else 'are'} not enabled.\n"
                    f"Enable {' and '.join(missing)} in config.yaml, or disable '{name}'."
                )

    @staticmethod
    def _topological_order(
        requirements: Mapping[str, Tuple[str, ...]], preferred: Iterable[str]
    ) -> List[str]:
        """Order modules so every dependency runs before its consumer.

        Ties are broken by config order, which keeps the pipeline deterministic
        and makes the ordering easy to reason about.
        """
        remaining = {name: set(deps) for name, deps in requirements.items()}
        order: List[str] = []
        preference = [n for n in preferred if n in requirements]

        while remaining:
            ready = [n for n in preference if n in remaining and not remaining[n]]
            if not ready:
                raise ModuleDependencyError(
                    "Circular module dependency detected among: "
                    f"{sorted(remaining)}. Module dependencies must form a DAG "
                    "(see ARCHITECTURE.md, 'No Circular Dependencies')."
                )
            chosen = ready[0]
            order.append(chosen)
            del remaining[chosen]
            for deps in remaining.values():
                deps.discard(chosen)

        return order


def register_module(module_cls: Type[BaseAIModule]) -> Type[BaseAIModule]:
    """Class decorator that registers an AI module.

    ::

        @register_module
        class PersonDetectionModule(BaseAIModule):
            module_name = "person_detection"
    """
    return ModuleRegistry.register(module_cls)
