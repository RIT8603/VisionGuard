"""Tests for live module reload: POST /api/modules -> ReloadController -> main.py.

Covers the new, narrow write path the dashboard gained: toggling which AI
modules are enabled without tearing down the dashboard server or editing
config.yaml by hand.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.server import build_app
from backend.main import _persist_enabled_modules, _with_enabled_modules
from backend.services.reload import ReloadController
from backend.services.state import DashboardState
from computer_vision.core.config import load_config
from computer_vision.modules.base import ModuleError
from computer_vision.modules.registry import ModuleRegistry

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


# --------------------------------------------------------------------------- #
# ReloadController
# --------------------------------------------------------------------------- #


class TestReloadController:
    def test_request_then_pop_returns_the_list_once(self):
        controller = ReloadController()
        assert controller.pop() is None  # nothing queued yet

        controller.request(["person_detection", "object_detection"])
        assert controller.pop() == ["person_detection", "object_detection"]
        assert controller.pop() is None  # consumed - a second pop is empty

    def test_a_later_request_replaces_an_unconsumed_one(self):
        controller = ReloadController()
        controller.request(["person_detection"])
        controller.request(["vehicle_detection"])  # last write wins
        assert controller.pop() == ["vehicle_detection"]

    def test_wait_returns_once_a_request_arrives(self):
        controller = ReloadController()
        controller.request([])
        controller.wait()  # must not block - the event is already set

    def test_fail_is_readable_back(self):
        controller = ReloadController()
        controller.request(["x"])
        controller.fail("module 'x' is unknown")
        assert controller.last_error == "module 'x' is unknown"
        # A fresh request clears a stale error from a previous attempt.
        controller.request(["person_detection"])
        assert controller.last_error is None


# --------------------------------------------------------------------------- #
# _with_enabled_modules
# --------------------------------------------------------------------------- #


class TestWithEnabledModules:
    def test_flips_only_the_requested_modules(self):
        config = load_config(str(CONFIG_PATH))
        new_config = _with_enabled_modules(config, ["person_detection", "face_detection"])
        assert set(new_config.enabled_module_names()) == {"person_detection", "face_detection"}

    def test_other_sections_are_untouched_by_identity(self):
        # A dump-and-revalidate round trip would mask dashboard.auth_token
        # (SecretStr) as "**********"; model_copy must not do that.
        config = load_config(str(CONFIG_PATH))
        new_config = _with_enabled_modules(config, ["person_detection"])
        assert new_config.dashboard is config.dashboard
        assert new_config.video is config.video
        assert new_config.processing is config.processing

    def test_module_params_besides_enabled_are_kept(self):
        config = load_config(str(CONFIG_PATH))
        new_config = _with_enabled_modules(config, ["person_detection"])
        assert (
            new_config.module_config("person_detection").params
            == config.module_config("person_detection").params
        )

    def test_a_name_with_no_existing_block_still_works(self):
        config = load_config(str(CONFIG_PATH))
        new_config = _with_enabled_modules(config, ["totally_new_module"])
        assert new_config.module_config("totally_new_module").enabled is True


# --------------------------------------------------------------------------- #
# _persist_enabled_modules
# --------------------------------------------------------------------------- #


class TestPersistEnabledModules:
    def test_round_trips_through_load_config(self, tmp_path: Path):
        target = tmp_path / "config.yaml"
        shutil.copy(CONFIG_PATH, target)

        _persist_enabled_modules(target, ["person_detection", "face_detection"])

        reloaded = load_config(str(target))
        assert set(reloaded.enabled_module_names()) == {"person_detection", "face_detection"}

    def test_only_the_boolean_tokens_change(self, tmp_path: Path):
        target = tmp_path / "config.yaml"
        shutil.copy(CONFIG_PATH, target)
        before = target.read_text(encoding="utf-8")

        _persist_enabled_modules(target, ["person_detection", "face_detection"])

        after = target.read_text(encoding="utf-8")
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        assert len(before_lines) == len(after_lines)
        changed = [
            (b, a) for b, a in zip(before_lines, after_lines) if b != a
        ]
        for before_line, after_line in changed:
            # Every change is exactly an `enabled:` boolean flip - nothing else
            # on the line (including a trailing comment) moved.
            assert "enabled:" in before_line and "enabled:" in after_line
            before_rest = before_line.split("enabled:", 1)[1].strip().split()[1:]
            after_rest = after_line.split("enabled:", 1)[1].strip().split()[1:]
            assert before_rest == after_rest

    def test_missing_file_does_not_raise(self, tmp_path: Path):
        _persist_enabled_modules(tmp_path / "does-not-exist.yaml", ["person_detection"])

    def test_unknown_module_names_in_the_file_are_left_alone(self, tmp_path: Path):
        target = tmp_path / "config.yaml"
        target.write_text(
            textwrap.dedent(
                """\
                modules:
                  person_detection:
                    enabled: false
                """
            ),
            encoding="utf-8",
        )
        # vehicle_detection has no block in this tiny file; must not crash or
        # add one.
        _persist_enabled_modules(target, ["person_detection", "vehicle_detection"])
        text = target.read_text(encoding="utf-8")
        assert "vehicle_detection" not in text
        assert "enabled: true" in text


# --------------------------------------------------------------------------- #
# POST /api/modules
# --------------------------------------------------------------------------- #


@pytest.fixture
def state() -> DashboardState:
    s = DashboardState()
    s.info = {
        "app": "VisionGuard AI",
        "version": "0.0.0",
        "source_type": "webcam",
        "source_id": "webcam-0",
        "resolution": "native",
        "modules": [],
    }
    return s


@pytest.fixture
def known_modules():
    return frozenset(
        {"person_detection", "person_tracking", "object_detection", "vehicle_detection"}
    )


class TestSetModulesEndpoint:
    def test_without_a_reload_controller_the_endpoint_is_disabled(self, state):
        client = TestClient(build_app(state), base_url="http://127.0.0.1")
        response = client.post("/api/modules", json={"enabled": ["person_detection"]})
        assert response.status_code == 503

    def test_valid_request_is_accepted_and_queued(self, state, known_modules):
        reload = ReloadController()
        app = build_app(state, reload=reload, known_modules=known_modules)
        client = TestClient(app, base_url="http://127.0.0.1")

        response = client.post(
            "/api/modules", json={"enabled": ["person_detection", "object_detection"]}
        )

        assert response.status_code == 202
        assert response.json() == {
            "status": "applying",
            "enabled": ["person_detection", "object_detection"],
        }
        assert reload.pop() == ["person_detection", "object_detection"]

    def test_unknown_module_name_is_rejected_before_queuing(self, state, known_modules):
        reload = ReloadController()
        app = build_app(state, reload=reload, known_modules=known_modules)
        client = TestClient(app, base_url="http://127.0.0.1")

        response = client.post(
            "/api/modules", json={"enabled": ["person_detection", "not_a_real_module"]}
        )

        assert response.status_code == 400
        assert "not_a_real_module" in response.json()["error"]
        assert reload.pop() is None  # the bad request never reached the queue

    @pytest.mark.parametrize("body", [{"enabled": "person_detection"}, {}, {"enabled": [1, 2]}])
    def test_malformed_body_is_rejected(self, state, known_modules, body):
        reload = ReloadController()
        app = build_app(state, reload=reload, known_modules=known_modules)
        client = TestClient(app, base_url="http://127.0.0.1")

        response = client.post("/api/modules", json=body)

        assert response.status_code == 400

    def test_security_headers_still_apply_to_the_write_endpoint(self, state, known_modules):
        reload = ReloadController()
        app = build_app(state, reload=reload, known_modules=known_modules)
        client = TestClient(app, base_url="http://127.0.0.1")

        response = client.post("/api/modules", json={"enabled": []})

        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'none'" in response.headers["content-security-policy"]

    def test_host_check_still_applies_to_the_write_endpoint(self, state, known_modules):
        from backend.api.server import AccessPolicy

        reload = ReloadController()
        policy = AccessPolicy.for_bind("127.0.0.1")
        app = build_app(state, policy=policy, reload=reload, known_modules=known_modules)
        client = TestClient(app, base_url="http://evil.example")

        response = client.post("/api/modules", json={"enabled": []})

        assert response.status_code == 400


# --------------------------------------------------------------------------- #
# DashboardState.resume
# --------------------------------------------------------------------------- #


class TestDashboardStateResume:
    def test_resume_flips_running_back_on(self):
        state = DashboardState()
        state.stop()
        assert state.running is False
        state.resume()
        assert state.running is True


# --------------------------------------------------------------------------- #
# End-to-end: a bad module selection raises ModuleError, as main.py expects
# --------------------------------------------------------------------------- #


def test_dependency_violation_raises_module_error_for_main_to_catch():
    config = load_config(str(CONFIG_PATH))
    bad_config = _with_enabled_modules(config, ["person_tracking"])  # no person_detection
    with pytest.raises(ModuleError):
        ModuleRegistry.build_enabled(bad_config)
