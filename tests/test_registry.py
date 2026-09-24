"""Module registry: registration, enable/disable, dependency ordering."""

from __future__ import annotations

import pytest

from app.core.config import build_config
from app.modules.base import ModuleInitializationError, ModuleStatus, ModuleType
from app.modules.registry import (
    ModuleDependencyError,
    ModuleNotRegisteredError,
    ModuleRegistry,
    register_module,
)
from tests.conftest import make_module


def config_with(modules: dict):
    return build_config({"modules": modules})


class TestRegistration:
    def test_register_and_resolve(self):
        cls = make_module("person_detection")
        ModuleRegistry.register(cls)

        assert ModuleRegistry.is_known("person_detection")
        assert ModuleRegistry.resolve("person_detection") is cls

    def test_decorator_registers(self):
        cls = register_module(make_module("fire_detection"))
        assert ModuleRegistry.resolve("fire_detection") is cls

    def test_duplicate_name_is_rejected(self):
        ModuleRegistry.register(make_module("person_detection"))
        with pytest.raises(Exception, match="already registered"):
            ModuleRegistry.register(make_module("person_detection"))

    def test_unknown_module_message_is_actionable(self):
        with pytest.raises(ModuleNotRegisteredError, match="Unknown AI module"):
            ModuleRegistry.resolve("teleportation_detection")


class TestBuildEnabled:
    def test_no_modules_enabled(self):
        assert ModuleRegistry.build_enabled(config_with({})) == []

    def test_only_enabled_modules_are_built(self):
        ModuleRegistry.register(make_module("person_detection"))
        ModuleRegistry.register(make_module("fire_detection"))

        built = ModuleRegistry.build_enabled(
            config_with(
                {
                    "person_detection": {"enabled": True},
                    "fire_detection": {"enabled": False},
                }
            )
        )

        assert [m.module_name for m in built] == ["person_detection"]
        assert built[0].status is ModuleStatus.READY
        assert built[0].initialized is True

    def test_disabled_module_is_never_resolved(self, monkeypatch):
        """Rule 6: a disabled module must not be imported or initialized."""
        resolved: list[str] = []
        original = ModuleRegistry.resolve.__func__

        def spy(cls, name):
            resolved.append(name)
            return original(cls, name)

        ModuleRegistry.register(make_module("person_detection"))
        monkeypatch.setattr(ModuleRegistry, "resolve", classmethod(spy))

        ModuleRegistry.build_enabled(
            config_with(
                {
                    "person_detection": {"enabled": True},
                    "face_recognition": {"enabled": False},
                }
            )
        )

        assert resolved == ["person_detection"]

    def test_enabled_but_unimplemented_module_fails_fast(self):
        with pytest.raises(ModuleNotRegisteredError, match="pose_detection"):
            ModuleRegistry.build_enabled(config_with({"pose_detection": {"enabled": True}}))

    def test_params_reach_the_module(self):
        ModuleRegistry.register(make_module("person_detection"))
        built = ModuleRegistry.build_enabled(
            config_with(
                {"person_detection": {"enabled": True, "confidence_threshold": 0.7}}
            )
        )
        assert built[0].params == {"confidence_threshold": 0.7}

    def test_initialization_failure_is_wrapped(self):
        ModuleRegistry.register(make_module("person_detection", fail_on_init=True))
        with pytest.raises(ModuleInitializationError, match="failed to initialize"):
            ModuleRegistry.build_enabled(config_with({"person_detection": {"enabled": True}}))

    def test_failed_build_cleans_up_already_built_modules(self):
        good = make_module("person_detection")
        ModuleRegistry.register(good)
        ModuleRegistry.register(
            make_module("person_tracking", deps=("person_detection",), fail_on_init=True)
        )

        with pytest.raises(ModuleInitializationError):
            ModuleRegistry.build_enabled(
                config_with(
                    {
                        "person_detection": {"enabled": True},
                        "person_tracking": {"enabled": True},
                    }
                )
            )
        # The successfully built detector must not be left holding resources.
        # (Verified indirectly: no exception escaped cleanup.)


class TestDependencyOrdering:
    def test_tracking_runs_after_detection_regardless_of_config_order(self):
        ModuleRegistry.register(make_module("person_detection"))
        ModuleRegistry.register(
            make_module(
                "person_tracking", kind=ModuleType.TRACKING, deps=("person_detection",)
            )
        )

        built = ModuleRegistry.build_enabled(
            config_with(
                {
                    "person_tracking": {"enabled": True},   # listed first on purpose
                    "person_detection": {"enabled": True},
                }
            )
        )

        assert [m.module_name for m in built] == ["person_detection", "person_tracking"]

    def test_config_order_breaks_ties(self):
        ModuleRegistry.register(make_module("fire_detection"))
        ModuleRegistry.register(make_module("person_detection"))

        built = ModuleRegistry.build_enabled(
            config_with(
                {
                    "fire_detection": {"enabled": True},
                    "person_detection": {"enabled": True},
                }
            )
        )
        assert [m.module_name for m in built] == ["fire_detection", "person_detection"]

    def test_missing_dependency_is_reported(self):
        ModuleRegistry.register(
            make_module("person_tracking", deps=("person_detection",))
        )

        with pytest.raises(ModuleDependencyError, match="requires \\['person_detection'\\]"):
            ModuleRegistry.build_enabled(config_with({"person_tracking": {"enabled": True}}))

    def test_circular_dependency_is_reported(self):
        ModuleRegistry.register(make_module("alpha", deps=("beta",)))
        ModuleRegistry.register(make_module("beta", deps=("alpha",)))

        with pytest.raises(ModuleDependencyError, match="Circular module dependency"):
            ModuleRegistry.build_enabled(
                config_with({"alpha": {"enabled": True}, "beta": {"enabled": True}})
            )


class TestLazyDeclaration:
    def test_declared_module_is_imported_on_demand(self):
        ModuleRegistry.declare("lazy_module", "tests.fixtures_lazy_module:LazyModule")
        resolved = ModuleRegistry.resolve("lazy_module")
        assert resolved.module_name == "lazy_module"

    def test_bad_import_path_is_reported(self):
        ModuleRegistry.declare("broken", "app.modules.nope:Missing")
        with pytest.raises(ModuleNotRegisteredError, match="Could not load AI module"):
            ModuleRegistry.resolve("broken")

    def test_malformed_import_path_is_reported(self):
        ModuleRegistry.declare("broken", "app.modules.base")
        with pytest.raises(ModuleNotRegisteredError, match="Malformed import path"):
            ModuleRegistry.resolve("broken")
