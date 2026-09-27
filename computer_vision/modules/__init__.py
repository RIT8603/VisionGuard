"""AI module layer.

Importing this package is cheap by design: it pulls in the interface and the
registry, never a concrete module or a model. Concrete modules are imported
lazily by :class:`~computer_vision.modules.registry.ModuleRegistry` and only when enabled.
"""

from computer_vision.modules.base import (
    BaseAIModule,
    ModuleError,
    ModuleInitializationError,
    ModuleResult,
    ModuleStatus,
    ModuleType,
)
from computer_vision.modules.registry import (
    ModuleDependencyError,
    ModuleNotRegisteredError,
    ModuleRegistry,
    register_module,
)

__all__ = [
    "BaseAIModule",
    "ModuleDependencyError",
    "ModuleError",
    "ModuleInitializationError",
    "ModuleNotRegisteredError",
    "ModuleRegistry",
    "ModuleResult",
    "ModuleStatus",
    "ModuleType",
    "register_module",
]
