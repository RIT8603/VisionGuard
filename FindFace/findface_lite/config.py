"""Pipeline settings.

Names and defaults deliberately mirror the FindFace Multi video-manager /
video-worker parameters (see docs/HOW_FINDFACE_TRACKS.md) so behaviour can be
compared one-to-one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Tuple

# (width, height, x, y) -- FindFace's "WxH+X+Y" geometry.
Rect = Tuple[int, int, int, int]


def parse_geometry(text: Optional[str]) -> Optional[Rect]:
    """Parse a FindFace-style 'WxH+X+Y' string."""
    if not text:
        return None
    size, _, offset = text.partition("+")
    w, h = size.lower().split("x")
    x, _, y = offset.partition("+")
    return int(w), int(h), int(x or 0), int(y or 0)


@dataclass
class Config:
    # --- motion detector (findface: imotion_threshold, --scale) -------------
    imotion_threshold: float = 0.002   # fraction of changed pixels; 0 disables
    motion_scale: float = 0.25         # downscale before differencing

    # --- detector ------------------------------------------------------------
    detector_model: str = "models/face_detection_yunet_2023mar.onnx"
    detector_score_threshold: float = 0.6
    detector_nms_threshold: float = 0.3
    resize_scale: float = 1.0          # findface worker: resize_scale

    # --- recognition (runs only when a watchlist is given) -----------------
    recognizer_model: str = "models/face_recognition_sface_2021dec.onnx"
    match_threshold: float = 0.363     # SFace cosine threshold for "same person"
    post_embedding: bool = False       # include the 128-d vector in events

    # --- filters (findface: filter_min_quality / filter_min_size / ...) -----
    filter_min_quality: float = 0.42   # 0..1, FindFace reference value is 0.42
    filter_min_size: int = 40          # bbox minor side, px
    filter_max_size: int = 8192        # bbox major side, px
    filter_max_angle: float = 45.0     # max estimated yaw/pitch, degrees
    roi: Optional[Rect] = None         # post only objects that entered ROI
    rot: Optional[Rect] = None         # detect/track only inside this region

    # --- tracker ---------------------------------------------------------------
    track_overlap_threshold: float = 0.25  # IoU to join an existing track
    track_miss_interval: float = 1.0       # s without detection -> close track
    track_max_duration_frames: int = 0     # 0 = unlimited
    track_min_length: int = 1              # frames before a track may post
    track_interpolate_bboxes: bool = True
    track_send_history: bool = False
    max_tracks: int = 64                   # fkvideo: --max-persons

    # --- best-snapshot policy -------------------------------------------------
    overall_only: bool = True                    # offline mode: 1 post/track
    realtime_post_interval: float = 1.0          # s
    realtime_post_every_interval: bool = False
    realtime_post_first_immediately: bool = False

    # --- output ----------------------------------------------------------------
    jpeg_quality: int = 95
    post_best_track_frame: bool = True     # save the full frame too
    post_first_track_frame: bool = False
    post_last_track_frame: bool = False
    crop_margin: float = 0.25              # extra context around the face crop

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        for key in ("roi", "rot"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = parse_geometry(kwargs[key])
            elif kwargs.get(key) is not None:
                kwargs[key] = tuple(kwargs[key])
        cfg = cls(**kwargs)
        cfg.extra = {k: v for k, v in data.items() if k not in known}
        return cfg
