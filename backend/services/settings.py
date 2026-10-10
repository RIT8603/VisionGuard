"""What the dashboard may change about a module's settings, and how.

``POST /api/modules`` turns modules on and off; this is the other half of
"configuration" from the browser: a module's tuning values - thresholds,
timers, severities. It is deliberately narrow:

* Only **scalar** settings already present in the module's config block are
  editable: booleans, numbers, and a ``severity``. Lists and maps (zones,
  label lists, sources) are not.
* Anything that points the platform at a file, a model, a device or a URL is
  never editable, whatever its type (see ``_LOCKED_WORDS``), so the dashboard
  cannot make the process load or open something new.
* A value must keep its type and be finite and not negative. Whether it is
  *sensible* is still the module's call: the change is applied by rebuilding
  the modules, and a value the module rejects in ``initialize`` leaves the
  previous setup running, exactly like a bad module selection.

Like the other writes, accepting a change only means it was queued.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Mapping, Tuple

from backend.services.reload import ReloadController
from computer_vision.core.config import ModuleConfig, VisionGuardConfig

SEVERITIES = ("info", "warning", "critical")

#: A key containing any of these is never editable from the dashboard.
_LOCKED_WORDS = (
    "model", "path", "url", "token", "device", "backend", "imgsz", "sha",
    "label", "source", "store", "file", "dir", "class",
)


def _locked(key: str) -> bool:
    lowered = key.lower()
    return any(word in lowered for word in _LOCKED_WORDS)


def _kind(key: str, value: Any) -> str | None:
    """``bool`` / ``int`` / ``float`` / ``severity``; None when not editable."""
    if _locked(key):
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if key == "severity" and isinstance(value, str) and value in SEVERITIES:
        return "severity"
    return None


def editable_settings(module_cfg: ModuleConfig) -> Dict[str, Dict[str, Any]]:
    """``{key: {"value": v, "kind": k}}`` for every editable setting of a module."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in module_cfg.params.items():
        kind = _kind(key, value)
        if kind is not None:
            out[key] = {"value": value, "kind": kind}
    return out


def coerce(kind: str, value: Any) -> Any:
    """``value`` checked against ``kind``. Raises ValueError with a readable reason."""
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("must be true or false")
        return value
    if kind == "severity":
        if value not in SEVERITIES:
            raise ValueError(f"must be one of {', '.join(SEVERITIES)}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("must be a finite number >= 0")
    if kind == "int":
        if float(value) != int(value):
            raise ValueError("must be a whole number")
        return int(value)
    return float(value)


def with_module_settings(
    config: VisionGuardConfig, changes: Mapping[str, Mapping[str, Any]]
) -> VisionGuardConfig:
    """A copy of ``config`` with ``changes`` (module -> {key: value}) applied."""
    modules = dict(config.modules)
    for name, params in changes.items():
        current = modules.get(name, ModuleConfig())
        merged = {"enabled": current.enabled, **current.params, **dict(params)}
        modules[name] = ModuleConfig(**merged)
    return config.model_copy(update={"modules": modules})


class SettingsManager:
    """Thread-safe view of the running module settings plus the one way to change them."""

    def __init__(self, reload: ReloadController | None, *, editable: bool = True) -> None:
        self._reload = reload
        self._editable = editable
        self._lock = threading.Lock()
        self._modules: Dict[str, ModuleConfig] = {}

    @property
    def editable(self) -> bool:
        return self._editable and self._reload is not None

    def set_current(self, config: VisionGuardConfig) -> None:
        """Record the settings now running (main loop, after every rebuild)."""
        with self._lock:
            self._modules = dict(config.modules)

    def listing(self) -> Dict[str, Any]:
        with self._lock:
            modules = dict(self._modules)
        return {
            "editable": self.editable,
            "apply_error": self._reload.last_error if self._reload is not None else None,
            "modules": {
                name: {"enabled": cfg.enabled, "settings": editable_settings(cfg)}
                for name, cfg in modules.items()
                if editable_settings(cfg)
            },
        }

    def submit(self, module: Any, params: Any) -> Tuple[int, Dict[str, Any]]:
        """Validate and queue a change. Returns ``(http_status, body)``."""
        if not self.editable:
            return 403, {"error": "Module settings are read-only for this run."}
        if not isinstance(module, str) or not isinstance(params, dict) or not params:
            return 400, {"error": "Send {'module': <name>, 'settings': {<key>: <value>, ...}}"}
        with self._lock:
            module_cfg = self._modules.get(module)
        if module_cfg is None:
            return 404, {"error": f"Unknown module: {module[:64]}"}

        allowed = editable_settings(module_cfg)
        clean: Dict[str, Any] = {}
        problems = []
        for key, value in params.items():
            spec = allowed.get(key) if isinstance(key, str) else None
            if spec is None:
                problems.append(f"{str(key)[:64]}: not an editable setting")
                continue
            try:
                clean[key] = coerce(spec["kind"], value)
            except ValueError as exc:
                problems.append(f"{key}: {exc}")
        if problems:
            return 400, {"error": "; ".join(problems)}

        assert self._reload is not None  # editable implies a controller
        self._reload.request_settings(module, clean)
        return 202, {"status": "applying", "module": module, "settings": clean}
