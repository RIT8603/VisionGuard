"""A module resolved through the registry's lazy-import path.

Exists to prove that ``ModuleRegistry.declare`` -> ``resolve`` actually imports
a Python module on demand, which is the mechanism that keeps a disabled
module's heavy dependencies out of the process.
"""

from __future__ import annotations

from computer_vision.core.context import FrameContext
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType


class LazyModule(BaseAIModule):
    module_name = "lazy_module"
    module_type = ModuleType.ANALYSIS
    version = "0.0.1"
    requires = ()

    def initialize(self) -> None:
        self.loaded = True

    def process(self, context: FrameContext) -> ModuleResult:
        return self.make_result(context, data={"lazy": True})
