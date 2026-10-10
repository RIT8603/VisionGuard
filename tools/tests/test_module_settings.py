"""Module settings from the dashboard: what is editable, validation, apply, persist."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.server import build_app
from backend.main import _persist_module_settings
from backend.services.reload import ReloadController
from backend.services.settings import SettingsManager, editable_settings, with_module_settings
from backend.services.state import DashboardState
from computer_vision.core.config import ModuleConfig, VisionGuardConfig, load_config


def config() -> VisionGuardConfig:
    return VisionGuardConfig(
        modules={
            "crowd_detection": ModuleConfig(enabled=True, threshold=5),
            "loitering": ModuleConfig(
                enabled=False,
                seconds=30,
                radius=0.08,
                severity="warning",
                sources=["person_tracking"],
                zones=[],
            ),
            "person_detection": ModuleConfig(
                enabled=True, model="m.pt", confidence_threshold=0.4, device="cpu"
            ),
        }
    )


def manager(editable: bool = True):
    reload = ReloadController()
    settings = SettingsManager(reload, editable=editable)
    settings.set_current(config())
    return reload, settings


class TestEditable:
    def test_only_scalars_and_never_models_paths_or_sources(self):
        cfg = config()
        assert set(editable_settings(cfg.modules["loitering"])) == {"seconds", "radius", "severity"}
        assert set(editable_settings(cfg.modules["person_detection"])) == {"confidence_threshold"}
        kinds = {k: v["kind"] for k, v in editable_settings(cfg.modules["loitering"]).items()}
        assert kinds == {"seconds": "int", "radius": "float", "severity": "severity"}


class TestSubmit:
    def test_valid_change_is_queued(self):
        reload, settings = manager()
        code, body = settings.submit("loitering", {"seconds": 45, "radius": 0.1, "severity": "critical"})
        assert code == 202
        assert reload.pop_settings() == {
            "loitering": {"seconds": 45, "radius": 0.1, "severity": "critical"}
        }
        assert reload.pop_settings() is None

    def test_changes_to_several_modules_accumulate(self):
        reload, settings = manager()
        settings.submit("loitering", {"seconds": 40})
        settings.submit("crowd_detection", {"threshold": 8})
        settings.submit("loitering", {"radius": 0.2})
        assert reload.pop_settings() == {
            "loitering": {"seconds": 40, "radius": 0.2},
            "crowd_detection": {"threshold": 8},
        }

    def test_rejections(self):
        reload, settings = manager()
        cases = [
            ("loitering", {"seconds": -1}),
            ("loitering", {"seconds": 2.5}),
            ("loitering", {"seconds": True}),
            ("loitering", {"seconds": "30"}),
            ("loitering", {"radius": float("inf")}),
            ("loitering", {"severity": "loud"}),
            ("loitering", {"zones": []}),
            ("loitering", {"sources": ["x"]}),
            ("person_detection", {"model": "evil.pt"}),
            ("loitering", {}),
        ]
        for module, params in cases:
            code, _ = settings.submit(module, params)
            assert code == 400, (module, params)
        assert settings.submit("nope", {"x": 1})[0] == 404
        assert reload.pop_settings() is None

    def test_read_only(self):
        _, settings = manager(editable=False)
        assert settings.submit("loitering", {"seconds": 40})[0] == 403
        assert settings.listing()["editable"] is False

    def test_apply_keeps_everything_else(self):
        cfg = with_module_settings(config(), {"loitering": {"seconds": 45}})
        loitering = cfg.modules["loitering"]
        assert loitering.params["seconds"] == 45
        assert loitering.params["sources"] == ["person_tracking"]
        assert loitering.enabled is False
        assert cfg.modules["crowd_detection"].params["threshold"] == 5


class TestApi:
    def test_list_and_post(self):
        reload, settings = manager()
        client = TestClient(build_app(DashboardState(), settings=settings), base_url="http://127.0.0.1")
        listing = client.get("/api/settings").json()
        assert listing["editable"] is True
        assert listing["modules"]["loitering"]["settings"]["seconds"] == {"value": 30, "kind": "int"}
        assert "model" not in listing["modules"]["person_detection"]["settings"]

        ok = client.post("/api/settings", json={"module": "loitering", "settings": {"seconds": 60}})
        assert ok.status_code == 202
        bad = client.post("/api/settings", json={"module": "loitering", "settings": {"zones": []}})
        assert bad.status_code == 400 and "zones" in bad.json()["error"]

    def test_without_manager(self):
        client = TestClient(build_app(DashboardState()), base_url="http://127.0.0.1")
        assert client.get("/api/settings").json()["modules"] == {}
        assert client.post("/api/settings", json={"module": "x", "settings": {"a": 1}}).status_code == 503


class TestPersist:
    def test_rewrites_values_in_place_keeping_comments(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "modules:\n"
            "  crowd_detection:\n"
            "    enabled: true\n"
            "    threshold: 5           # people at or above this count = crowded\n"
            "\n"
            "  # ---- Behaviour ----\n"
            "  loitering:\n"
            "    enabled: false\n"
            "    seconds: 30\n"
            "    severity: warning\n"
            "  intrusion_detection:\n"
            "    seconds: 30\n",
            encoding="utf-8",
        )
        _persist_module_settings(
            path, {"crowd_detection": {"threshold": 8}, "loitering": {"seconds": 45, "severity": "critical", "missing": 1}}
        )
        text = path.read_text(encoding="utf-8")
        assert "    threshold: 8           # people at or above this count = crowded\n" in text
        assert "    seconds: 45\n    severity: critical\n" in text
        assert text.endswith("  intrusion_detection:\n    seconds: 30\n")
        assert "missing" not in text
        assert load_config(path, use_env=False).modules["crowd_detection"].params["threshold"] == 8

    def test_real_config_round_trips(self, tmp_path):
        source = load_config("configs/config.yaml", use_env=False)
        path = tmp_path / "config.yaml"
        path.write_text(open("configs/config.yaml", encoding="utf-8").read(), encoding="utf-8")
        _persist_module_settings(path, {"crowd_detection": {"threshold": 9}})
        after = load_config(path, use_env=False)
        assert after.modules["crowd_detection"].params["threshold"] == 9
        assert set(after.modules) == set(source.modules)


class TestMainEndToEnd:
    def test_dashboard_changes_a_setting_live_and_saves_it(self, tmp_path):
        import threading
        import time

        import httpx
        import yaml

        from backend.main import main
        from backend.services.cameras import cameras_file_for, save_camera_file
        from computer_vision.core.config import CameraConfig
        from .test_multi_camera import _config, _free_port, write_clip

        clip = write_clip(tmp_path / "loop.avi", 30)
        config = _config(
            tmp_path,
            "  crowd_detection:\n                enabled: true\n                threshold: 5  # people\n",
        )
        save_camera_file(
            cameras_file_for(config), [CameraConfig(id="one", source="file", path=str(clip), loop=True)]
        )
        port = _free_port()
        worker = threading.Thread(
            target=lambda: main(["--config", str(config), "--no-window", "--dashboard",
                                 "--dashboard-port", str(port), "--duration", "20"]),
            daemon=True,
        )
        worker.start()
        base = f"http://127.0.0.1:{port}"

        def wait_for(predicate, seconds=15):
            deadline = time.monotonic() + seconds
            last = None
            while time.monotonic() < deadline:
                try:
                    last = httpx.get(f"{base}/api/settings", timeout=2).json()
                    if predicate(last):
                        return last
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            raise AssertionError(f"timed out; last: {last}")

        wait_for(lambda b: b["modules"].get("crowd_detection", {}).get("settings", {}).get("threshold", {}).get("value") == 5)
        response = httpx.post(f"{base}/api/settings", json={"module": "crowd_detection", "settings": {"threshold": 7}})
        assert response.status_code == 202
        wait_for(lambda b: b["modules"]["crowd_detection"]["settings"]["threshold"]["value"] == 7)
        saved = yaml.safe_load(config.read_text())
        assert saved["modules"]["crowd_detection"]["threshold"] == 7
        assert "# people" in config.read_text()

        # A value the module rejects keeps the previous setup and says why.
        httpx.post(f"{base}/api/settings", json={"module": "crowd_detection", "settings": {"threshold": 0}})
        body = wait_for(lambda b: b["apply_error"] is not None)
        assert "threshold" in body["apply_error"]
        assert body["modules"]["crowd_detection"]["settings"]["threshold"]["value"] == 7
