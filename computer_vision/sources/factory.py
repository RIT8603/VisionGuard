"""Video source factory.

This is the *only* place in VisionGuard that knows which concrete source class
corresponds to which configuration value. The Vision Core, the pipeline and
every AI module see nothing but :class:`~computer_vision.sources.base.BaseVideoSource`.
"""

from __future__ import annotations

from typing import Callable, Dict

from computer_vision.core.config import VideoConfig, VideoSourceType
from computer_vision.sources.base import BaseVideoSource
from computer_vision.sources.rtsp import RTSPSource
from computer_vision.sources.video_file import VideoFileSource
from computer_vision.sources.webcam import WebcamSource
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

SourceBuilder = Callable[[VideoConfig, str | None], BaseVideoSource]

_BUILDERS: Dict[VideoSourceType, SourceBuilder] = {
    VideoSourceType.WEBCAM: lambda cfg, sid: WebcamSource(cfg, sid),
    VideoSourceType.FILE: lambda cfg, sid: VideoFileSource(cfg, sid),
    VideoSourceType.RTSP: lambda cfg, sid: RTSPSource(cfg, sid),
}


def create_video_source(
    config: VideoConfig, source_id: str | None = None
) -> BaseVideoSource:
    """Build the video source described by ``config``.

    Args:
        config: The validated ``video:`` section.
        source_id: Optional override for the id stamped on every frame.

    Returns:
        An unopened :class:`BaseVideoSource`.

    Raises:
        NotImplementedError: for source types not yet delivered (file, rtsp).
        ValueError: for an unknown source type.
    """
    builder = _BUILDERS.get(config.source)
    if builder is None:  # pragma: no cover - guarded by the config enum
        raise ValueError(
            f"Unknown video source type: {config.source!r}. "
            f"Supported: {[t.value for t in VideoSourceType]}"
        )

    source = builder(config, source_id)
    logger.debug("Created %s for source type '%s'", type(source).__name__, config.source.value)
    return source


def available_source_types() -> Dict[str, bool]:
    """Map of source type -> implemented. Used by ``--list-sources``."""
    return {
        VideoSourceType.WEBCAM.value: True,
        VideoSourceType.FILE.value: True,
        VideoSourceType.RTSP.value: True,
    }
