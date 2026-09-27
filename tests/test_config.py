"""Configuration loading, overriding and validation."""

from __future__ import annotations

import textwrap

import pytest

from computer_vision.core.config import (
    ConfigError,
    VideoSourceType,
    VisionGuardConfig,
    build_config,
    load_config,
)


def write_config(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_defaults_apply_to_empty_config(tmp_path):
    path = write_config(tmp_path, "{}")
    config = load_config(path, use_env=False)

    assert config.app.name == "VisionGuard AI"
    assert config.video.source is VideoSourceType.WEBCAM
    assert config.video.device_id == 0
    assert config.processing.show_window is True
    assert config.modules == {}


def test_yaml_values_are_loaded(tmp_path):
    path = write_config(
        tmp_path,
        """
        app:
          log_level: debug
        video:
          source: webcam
          device_id: 2
          width: 640
          height: 480
        processing:
          target_width: 320
          show_window: false
        modules:
          person_detection:
            enabled: true
            confidence_threshold: 0.6
          fire_detection:
            enabled: false
        """,
    )
    config = load_config(path, use_env=False)

    assert config.app.log_level == "DEBUG"  # normalised to upper case
    assert config.video.device_id == 2
    assert config.processing.target_width == 320
    assert config.enabled_module_names() == ["person_detection"]
    # Extra keys become module params; the core never interprets them.
    assert config.modules["person_detection"].params == {"confidence_threshold": 0.6}


def test_env_overrides_yaml(tmp_path, monkeypatch):
    path = write_config(tmp_path, "video:\n  device_id: 0\n")
    monkeypatch.setenv("VIDEO_DEVICE_ID", "3")
    monkeypatch.setenv("LOG_LEVEL", "warning")

    config = load_config(path, use_env=True)

    assert config.video.device_id == 3
    assert config.app.log_level == "WARNING"


def test_prefixed_env_wins_over_bare(tmp_path, monkeypatch):
    path = write_config(tmp_path, "{}")
    monkeypatch.setenv("VIDEO_DEVICE_ID", "1")
    monkeypatch.setenv("VISIONGUARD_VIDEO_DEVICE_ID", "7")

    assert load_config(path, use_env=True).video.device_id == 7


def test_cli_overrides_win(tmp_path, monkeypatch):
    path = write_config(tmp_path, "video:\n  device_id: 0\n")
    monkeypatch.setenv("VIDEO_DEVICE_ID", "3")

    config = load_config(path, use_env=True, overrides={"video": {"device_id": 5}})

    assert config.video.device_id == 5


def test_none_overrides_are_ignored(tmp_path):
    path = write_config(tmp_path, "video:\n  device_id: 4\n")

    config = load_config(path, use_env=False, overrides={"video": {"device_id": None}})

    assert config.video.device_id == 4


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "nope.yaml", use_env=False)


def test_invalid_yaml_is_a_clear_error(tmp_path):
    path = write_config(tmp_path, "video:\n  - [unclosed\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path, use_env=False)


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="extra_inputs_are_not_permitted|Extra inputs"):
        build_config({"video": {"definitely_not_a_field": 1}})


def test_rtsp_requires_url():
    with pytest.raises(ConfigError, match="rtsp_url is required"):
        build_config({"video": {"source": "rtsp"}})


def test_file_source_requires_path():
    with pytest.raises(ConfigError, match="path is required"):
        build_config({"video": {"source": "file"}})


def test_bad_log_level_is_rejected():
    with pytest.raises(ConfigError, match="log_level"):
        build_config({"app": {"log_level": "CHATTY"}})


def test_negative_device_id_is_rejected():
    with pytest.raises(ConfigError, match="device_id"):
        build_config({"video": {"device_id": -1}})


def test_zero_target_width_is_rejected():
    with pytest.raises(ConfigError, match="target_width"):
        build_config({"processing": {"target_width": 0}})


def test_null_target_width_is_allowed():
    config = build_config({"processing": {"target_width": None}})
    assert config.processing.target_width is None


def test_resolved_source_id():
    assert build_config({"video": {"device_id": 1}}).video.resolved_source_id() == "webcam-1"
    explicit = build_config({"video": {"source_id": "lobby-cam"}})
    assert explicit.video.resolved_source_id() == "lobby-cam"


def test_shipped_config_is_valid():
    """configs/config.yaml must always load - it is the documented entry point."""
    config = load_config("configs/config.yaml", use_env=False)

    assert isinstance(config, VisionGuardConfig)
    assert config.video.source is VideoSourceType.WEBCAM


def test_shipped_config_only_enables_implemented_modules():
    """Guards against shipping a config that fails on startup.

    Every module enabled in configs/config.yaml must be resolvable by the
    registry, so cloning the repo and running it works without edits.
    """
    from computer_vision.modules.registry import ModuleRegistry

    config = load_config("configs/config.yaml", use_env=False)
    unknown = [
        name for name in config.enabled_module_names() if not ModuleRegistry.is_known(name)
    ]

    assert unknown == [], f"config.yaml enables unimplemented modules: {unknown}"
