"""Choosing the PyTorch device: ``auto`` follows the GPU, a missing GPU never crashes."""

from __future__ import annotations

import pytest

from computer_vision.inference import device
from computer_vision.inference.shared import share_key


@pytest.fixture(autouse=True)
def fresh_cache():
    device._resolve.cache_clear()
    yield
    device._resolve.cache_clear()


def gpu(monkeypatch, present: bool) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: present)
    if present:
        import torch

        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i=0: "Test GPU", raising=False)


def test_auto_uses_the_gpu_when_there_is_one(monkeypatch):
    gpu(monkeypatch, True)
    assert device.resolve_torch_device("auto") == "cuda:0"


def test_auto_uses_the_cpu_when_there_is_none(monkeypatch):
    gpu(monkeypatch, False)
    assert device.resolve_torch_device("auto") == "cpu"


@pytest.mark.parametrize("asked", ["cuda", "gpu", "0", "cuda:0"])
def test_explicit_gpu_requests(monkeypatch, asked):
    gpu(monkeypatch, True)
    assert device.resolve_torch_device(asked) == "cuda:0"


def test_a_missing_gpu_falls_back_to_cpu_with_a_warning(monkeypatch, caplog):
    gpu(monkeypatch, False)
    with caplog.at_level("WARNING"):
        assert device.resolve_torch_device("cuda") == "cpu"
    assert "using the CPU" in caplog.text


def test_cpu_and_unknown_devices_pass_through(monkeypatch):
    gpu(monkeypatch, True)
    assert device.resolve_torch_device("cpu") == "cpu"
    assert device.resolve_torch_device("mps") == "mps"
    assert device.resolve_torch_device(None) == "cpu"


def test_detectors_spelling_the_gpu_differently_still_share_a_model(monkeypatch):
    gpu(monkeypatch, True)
    a = share_key("ultralytics", {"model": "m.pt", "device": "auto"})
    b = share_key("ultralytics", {"model": "m.pt", "device": "cuda:0"})
    assert a == b


def test_cpu_and_gpu_are_different_models(monkeypatch):
    gpu(monkeypatch, True)
    assert share_key("ultralytics", {"model": "m.pt", "device": "cpu"}) != share_key(
        "ultralytics", {"model": "m.pt", "device": "auto"}
    )
