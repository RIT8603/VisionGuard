"""Deployment preflight regressions; no package install, models or servers run."""
import hashlib
from pathlib import Path

import pytest
import yaml

from deploy.prepare_backend import configure_devices, prepare, validate_runtime
from computer_vision.inference.base import ModelLoadError


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("video.mp4").write_bytes(b"asset-presence-fixture-not-a-CV-test")
    weights = Path("detector.pt")
    weights.write_bytes(b"weight-integrity-fixture-not-a-model")
    raw = {
        "video": {"source": "webcam"},
        "modules": {
            "person_detection": {"enabled": True, "backend": "ultralytics", "model": str(weights),
                                 "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()},
            "person_tracking": {"enabled": True},
            "low_light": {"enabled": True, "threshold": 37},
        },
    }
    source = Path("source.yaml")
    source.write_text(yaml.safe_dump(raw))
    return source, Path("runtime")


def test_first_boot_is_separate_and_explicitly_limited_to_detection_tracking(deployment):
    source, output = deployment
    original = source.read_bytes()
    result = prepare("cpu", source, output, "video.mp4")
    raw = yaml.safe_load((output / "config.yaml").read_text())
    assert source.read_bytes() == original
    assert result["modules"] == ["person_detection", "person_tracking"]
    assert raw["modules"]["person_detection"]["device"] == "cpu"
    assert raw["modules"]["low_light"]["enabled"] is False
    assert raw["video"]["loop"] is True
    assert raw["dashboard"]["max_streams"] >= 5
    assert raw["dashboard"]["auth_token"] is None


def test_gpu_migration_preserves_manual_module_settings_and_cameras(deployment):
    source, output = deployment
    prepare("cpu", source, output, "video.mp4")
    target = output / "config.yaml"
    raw = yaml.safe_load(target.read_text())
    raw["modules"]["low_light"]["enabled"] = True
    target.write_text(yaml.safe_dump(raw))
    cameras = output / "cameras.yaml"
    cameras.write_text(yaml.safe_dump({"cameras": [
        {"id": "cam-2", "source": "file", "path": "video.mp4", "loop": True,
         "modules": ["person_detection", "person_tracking"]}]}))
    before = cameras.read_bytes()
    result = prepare("gpu", source, output, None)
    raw = yaml.safe_load(target.read_text())
    assert result["preserved_existing_selection"] is True
    assert raw["modules"]["low_light"]["enabled"] is True
    assert raw["modules"]["low_light"]["threshold"] == 37
    assert raw["modules"]["person_detection"]["device"] == "cuda:0"
    assert cameras.read_bytes() == before


def test_per_camera_dependencies_cannot_be_satisfied_by_another_camera(deployment):
    source, output = deployment
    prepare("cpu", source, output, "video.mp4")
    (output / "cameras.yaml").write_text(yaml.safe_dump({"cameras": [
        {"id": "cam-1", "source": "file", "path": "video.mp4", "loop": True,
         "modules": ["person_detection"]},
        {"id": "cam-2", "source": "file", "path": "video.mp4", "loop": True,
         "modules": ["person_tracking"]},
    ]}))
    with pytest.raises(ValueError, match="missing enabled dependencies on a camera"):
        validate_runtime(output / "config.yaml")


def test_failed_preflight_keeps_last_config(deployment):
    source, output = deployment
    prepare("cpu", source, output, "video.mp4")
    original = (output / "config.yaml").read_bytes()
    Path("detector.pt").write_bytes(b"wrong-model")
    with pytest.raises(ModelLoadError, match="integrity check"):
        prepare("gpu", source, output, None)
    assert (output / "config.yaml").read_bytes() == original
    assert sorted(p.name for p in output.iterdir()) == ["config.yaml"]


@pytest.mark.parametrize("path", [r"C:\Users\user\video.mp4", "D:/video.mp4", "missing.mp4"])
def test_rejects_windows_or_missing_video_paths(deployment, path):
    source, output = deployment
    with pytest.raises(ValueError):
        prepare("cpu", source, output, path)
    assert not (output / "config.yaml").exists()


def test_rejects_non_looping_demo_and_ec2_webcam(deployment):
    source, output = deployment
    prepare("cpu", source, output, "video.mp4")
    target = output / "config.yaml"
    raw = yaml.safe_load(target.read_text())
    raw["video"]["loop"] = False
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="loop: true"):
        validate_runtime(target)
    raw["video"]["source"] = "webcam"
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="no local webcam"):
        validate_runtime(target)


def test_mode_change_keeps_openvino_adapters_off_nvidia_cuda():
    modules = {
        "detector": {"backend": "ultralytics", "models": [{"model": "rifle.pt"}]},
        "intel": {"backend": "openvino", "device": "auto"},
        "person_reid": {"model": "reid.onnx", "gallery_path": "gallery.npz"},
        "analysis": {"enabled": False},
    }
    configure_devices(modules, "gpu")
    assert modules["detector"]["device"] == "cuda:0"
    assert modules["detector"]["models"][0]["device"] == "cuda:0"
    assert modules["intel"]["device"] == modules["person_reid"]["device"] == "cpu"
    assert "device" not in modules["analysis"]
