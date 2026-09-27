"""Video source abstraction.

The rest of VisionGuard only ever talks to :class:`BaseVideoSource`. Swapping a
webcam for an RTSP camera is a configuration change, not a code change.
"""

from computer_vision.sources.base import (
    BaseVideoSource,
    FramePacket,
    FrameReadError,
    SourceInfo,
    SourceNotOpenError,
    SourceOpenError,
    VideoSourceError,
)
from computer_vision.sources.factory import create_video_source
from computer_vision.sources.webcam import WebcamSource

__all__ = [
    "BaseVideoSource",
    "FramePacket",
    "FrameReadError",
    "SourceInfo",
    "SourceNotOpenError",
    "SourceOpenError",
    "VideoSourceError",
    "WebcamSource",
    "create_video_source",
]
