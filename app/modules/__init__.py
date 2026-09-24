"""AI module layer.

Importing this package is cheap by design: it pulls in the interface and the
registry, never a concrete module or a model. Concrete modules are imported
lazily by :class:`~app.modules.registry.ModuleRegistry` and only when enabled.
"""

from app.modules.base import (
    BaseAIModule,
    ModuleError,
    ModuleInitializationError,
    ModuleResult,
    ModuleStatus,
    ModuleType,
)
from app.modules.registry import (
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
