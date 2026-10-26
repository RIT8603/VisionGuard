"""findface-lite: FindFace-style face detection, tracking and best-snapshot posting."""
from .config import Config
from .pipeline import FacePipeline, run_video
from .recognition import FaceEmbedder, Recognizer, Watchlist
from .sinks import CallbackSink, DirectorySink, WebhookSink

__all__ = ["Config", "FacePipeline", "run_video", "DirectorySink", "WebhookSink", "CallbackSink",
           "FaceEmbedder", "Watchlist", "Recognizer"]
