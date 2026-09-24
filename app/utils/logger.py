"""Structured logging setup for VisionGuard AI.

Call :func:`setup_logging` once at application start. Every other module uses
:func:`get_logger` and never configures handlers itself.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger.

    Args:
        level: One of DEBUG / INFO / WARNING / ERROR / CRITICAL.
        log_file: Optional path; when set, logs are mirrored to this file.
    """
    global _configured

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove handlers we installed previously so repeated calls stay idempotent.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)


def is_configured() -> bool:
    return _configured
