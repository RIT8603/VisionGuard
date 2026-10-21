"""Choosing the PyTorch device (CPU or an NVIDIA GPU) for a model.

``device: auto`` picks the GPU when this machine has a usable one and the CPU
otherwise, so one config runs unchanged on a laptop and on a GPU box. A GPU
that was asked for by name but is not there (``cuda``, ``0``, ``gpu``) falls
back to the CPU with a warning rather than failing the whole run.

PyTorch is imported lazily: an OpenVINO-only install has none, and that must
simply mean "CPU", not an ImportError.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

_GPU_WORDS = {"cuda", "gpu"}


def cuda_available() -> bool:
    """Whether PyTorch is installed *and* can see an NVIDIA GPU."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver is "no GPU", not a crash
        return False


@lru_cache(maxsize=None)
def _resolve(requested: str) -> str:
    wants_gpu = requested in _GPU_WORDS or requested.startswith("cuda:") or requested.isdigit()

    if requested == "auto":
        if cuda_available():
            import torch

            logger.info("device=auto: using the GPU (%s)", torch.cuda.get_device_name(0))
            return "cuda:0"
        logger.info("device=auto: no usable NVIDIA GPU found, using the CPU")
        return "cpu"

    if wants_gpu:
        if not cuda_available():
            logger.warning(
                "device '%s' was requested but PyTorch cannot see a GPU "
                "(CPU-only PyTorch build, missing driver, or no NVIDIA GPU) - using the CPU",
                requested,
            )
            return "cpu"
        if requested in _GPU_WORDS:
            return "cuda:0"
        return f"cuda:{requested}" if requested.isdigit() else requested

    return requested  # cpu, mps, ...


def resolve_torch_device(requested: Any) -> str:
    """The concrete device string for ``requested`` (``auto``, ``cpu``, ``cuda``, ``0`` ...)."""
    text = str(requested if requested is not None else "cpu").strip().lower() or "cpu"
    return _resolve(text)
