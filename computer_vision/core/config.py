"""Configuration loading and validation.

Precedence (lowest to highest):

    1. Defaults declared on the Pydantic models below
    2. ``configs/config.yaml`` (or the path given to :func:`load_config`)
    3. Environment variables / ``.env``
    4. Command-line flags (applied by ``backend.main``)

Every behavioural switch in VisionGuard is expressed here. No module, source or
pipeline component is allowed to read ``os.environ`` directly.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
ENV_PREFIX = "VISIONGUARD_"

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(Exception):
    """Raised when configuration is missing, malformed or internally invalid."""


class VideoSourceType(str, Enum):
    """The supported video input kinds.

    The rest of the pipeline never branches on this value — only
    ``computer_vision.sources.factory`` does.
    """

    WEBCAM = "webcam"
    FILE = "file"
    RTSP = "rtsp"
    YOUTUBE = "youtube"


class CaptureBackend(str, Enum):
    """OpenCV capture backend selection.

    ``AUTO`` picks MSMF on Windows, V4L2 on Linux, and lets OpenCV decide
    elsewhere. See :class:`~computer_vision.sources.webcam.WebcamSource` for why MSMF wins
    on Windows despite its slower open.
    """

    AUTO = "auto"
    ANY = "any"
    DSHOW = "dshow"
    MSMF = "msmf"
    V4L2 = "v4l2"
    FFMPEG = "ffmpeg"
    GSTREAMER = "gstreamer"


class AppConfig(BaseModel):
    """Application-wide settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "VisionGuard AI"
    log_level: str = "INFO"
    log_file: str | None = None

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return upper


class VideoConfig(BaseModel):
    """Video input settings shared by every source implementation.

    Fields not relevant to the selected ``source`` are simply ignored, which is
    what lets a single config file describe webcam, file and RTSP deployments.
    """

    model_config = ConfigDict(extra="forbid")

    source: VideoSourceType = VideoSourceType.WEBCAM
    source_id: str | None = Field(
        default=None,
        description="Stable identifier stamped onto every frame. Auto-derived when unset.",
    )
    backend: CaptureBackend = CaptureBackend.AUTO

    # Webcam
    device_id: int = 0
    fourcc: str | None = Field(
        default=None,
        description=(
            "Pixel format to request from the camera, e.g. 'MJPG'. Left unset "
            "by default: on some cameras MJPG lifts the frame-rate cap, on "
            "others the driver ignores or is slowed by it, so it is opt-in "
            "after measuring rather than a blanket default."
        ),
    )

    # Requested capture properties (a driver may ignore these).
    width: int = 1280
    height: int = 720
    fps: int = 30

    # File source
    path: str | None = None
    loop: bool = False
    start_frame: int = Field(
        default=0,
        description="Frame to start playback from. Useful for evaluating a segment.",
    )
    playback_speed: float = Field(
        default=0.0,
        description=(
            "Multiple of the file's native frame rate to play at. 0 means "
            "unthrottled (process as fast as possible), which is what you want "
            "for benchmarking; 1.0 plays at real time."
        ),
    )

    # RTSP source
    rtsp_url: str | None = None
    rtsp_transport: str = Field(
        default="tcp",
        description=(
            "'tcp' is reliable and the right default; 'udp' has lower latency "
            "but drops packets, producing smeared frames on a busy network."
        ),
    )

    # YouTube source
    youtube_url: str | None = None

    # Resilience
    open_timeout_sec: float = 10.0
    read_timeout_sec: float = Field(
        default=5.0,
        description="How long a single frame read may block before it is treated as a stall.",
    )
    read_retries: int = 5
    read_retry_delay_sec: float = 0.05
    reconnect_attempts: int = Field(
        default=5,
        description="Consecutive reconnects allowed before the source gives up. 0 disables reconnect.",
    )
    reconnect_delay_sec: float = Field(
        default=2.0,
        description="Base delay between reconnects; doubles each consecutive failure.",
    )
    reconnect_max_delay_sec: float = 30.0

    @field_validator("rtsp_transport")
    @classmethod
    def _validate_transport(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        return lowered

    @field_validator("device_id")
    @classmethod
    def _validate_device(cls, value: int) -> int:
        if value < 0:
            raise ValueError("device_id must be >= 0")
        return value

    @field_validator("width", "height")
    @classmethod
    def _validate_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("width/height must be positive")
        return value

    @field_validator("fps")
    @classmethod
    def _validate_fps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("fps must be positive")
        return value

    @field_validator("start_frame")
    @classmethod
    def _validate_start_frame(cls, value: int) -> int:
        if value < 0:
            raise ValueError("start_frame must be >= 0")
        return value

    @field_validator("playback_speed")
    @classmethod
    def _validate_playback_speed(cls, value: float) -> float:
        if value < 0:
            raise ValueError("playback_speed must be >= 0 (0 = unthrottled)")
        return value

    @field_validator("fourcc")
    @classmethod
    def _validate_fourcc(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("fourcc must be exactly 4 characters, e.g. 'MJPG'")
        return value.upper()

    @field_validator(
        "open_timeout_sec",
        "read_timeout_sec",
        "read_retry_delay_sec",
        "reconnect_delay_sec",
        "reconnect_max_delay_sec",
        "read_retries",
        "reconnect_attempts",
    )
    @classmethod
    def _validate_non_negative(cls, value, info):
        # A negative timeout/delay/count has no meaning and was silently
        # accepted before (e.g. a negative sleep raises mid-reconnect).
        if value < 0:
            raise ValueError(f"video.{info.field_name} must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_source_requirements(self) -> "VideoConfig":
        if self.source is VideoSourceType.FILE and not self.path:
            raise ValueError("video.path is required when video.source is 'file'")
        if self.source is VideoSourceType.RTSP and not self.rtsp_url:
            raise ValueError("video.rtsp_url is required when video.source is 'rtsp'")
        if self.source is VideoSourceType.YOUTUBE and not self.youtube_url:
            raise ValueError("video.youtube_url is required when video.source is 'youtube'")
        return self

    def resolved_source_id(self) -> str:
        """A human-readable, stable id for this source."""
        if self.source_id:
            return self.source_id
        if self.source is VideoSourceType.WEBCAM:
            return f"webcam-{self.device_id}"
        if self.source is VideoSourceType.FILE:
            return f"file-{Path(self.path or 'unknown').stem}"
        if self.source is VideoSourceType.YOUTUBE:
            return "youtube-0"
        return "rtsp-0"


#: Camera ids are used in URLs and file names, so keep them boring.
_CAMERA_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_RTSP_SCHEMES = ("rtsp", "rtsps")
_YOUTUBE_URL_REGEX = re.compile(
    r"^https?://(www\.|m\.)?(youtube\.com/(watch\?v=|live/|embed/|shorts/)|youtu\.be/)[a-zA-Z0-9_\-]+",
    re.IGNORECASE,
)


def is_youtube_url(url: str | None) -> bool:
    """True if ``url`` looks like a standard YouTube watch, live, or short link."""
    if not url:
        return False
    return bool(_YOUTUBE_URL_REGEX.match(url.strip()))


class CameraConfig(BaseModel):
    """One camera in a multi-camera run.

    Only what differs between cameras lives here. Everything else (capture
    size, RTSP transport, timeouts, reconnect budget) comes from the shared
    ``video:`` section through :meth:`to_video`, so a camera entry stays as
    short as ``{id: gate, source: rtsp, rtsp_url: rtsp://...}``.

    The stored list lives in ``configs/cameras.yaml`` (see
    ``backend.services.cameras``): an RTSP URL usually carries a password,
    which does not belong in the committed ``config.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable id: lower-case letters, digits, '-' and '_' (max 32).")
    name: str | None = Field(default=None, description="Display name; defaults to the id.")
    source: VideoSourceType = VideoSourceType.RTSP
    device_id: int = 0
    rtsp_url: str | None = None
    youtube_url: str | None = None
    path: str | None = None
    loop: bool = False
    modules: List[str] | None = Field(
        default=None,
        description=(
            "AI modules this camera runs, dependencies included. Omit to follow "
            "the shared selection (the `modules:` section / the Modules page); "
            "an empty list means video only."
        ),
    )

    @field_validator("modules")
    @classmethod
    def _validate_modules(cls, value: List[str] | None) -> List[str] | None:
        if value is None:
            return None
        names: List[str] = []
        for item in value:
            name = item.strip()
            if not name or len(name) > 64:
                raise ValueError("module names must be 1-64 characters")
            if name not in names:
                names.append(name)
        if len(names) > 64:
            raise ValueError("at most 64 modules per camera")
        return names

    @field_validator("id", mode="before")
    @classmethod
    def _validate_id(cls, value):
        text = str(value).strip().lower()
        if not _CAMERA_ID.match(text):
            raise ValueError(
                "camera id must be 1-32 characters: lower-case letters, digits, '-' or '_', "
                "starting with a letter or digit"
            )
        return text

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if len(value) > 60:
            raise ValueError("camera name must be at most 60 characters")
        return value or None

    @field_validator("device_id")
    @classmethod
    def _validate_device_id(cls, value: int) -> int:
        if value < 0:
            raise ValueError("device_id must be >= 0")
        return value

    @field_validator("rtsp_url", "path", "youtube_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_source(self) -> "CameraConfig":
        if self.source is VideoSourceType.RTSP:
            if not self.rtsp_url:
                raise ValueError("rtsp_url is required for an rtsp camera")
            # ffmpeg understands far more than RTSP (file:, concat:, http:, ...).
            # A camera entry can come from the dashboard, so only RTSP is allowed.
            scheme = self.rtsp_url.split(":", 1)[0].lower()
            if scheme not in _RTSP_SCHEMES or "://" not in self.rtsp_url:
                raise ValueError("rtsp_url must start with rtsp:// or rtsps://")
        elif self.source is VideoSourceType.FILE and not self.path:
            raise ValueError("path is required for a file camera")
        elif self.source is VideoSourceType.YOUTUBE:
            if not self.youtube_url:
                raise ValueError("youtube_url is required for a youtube camera")
            if not is_youtube_url(self.youtube_url):
                raise ValueError(
                    "youtube_url must be a valid YouTube URL (e.g. https://www.youtube.com/watch?v=... or https://www.youtube.com/live/... or https://youtu.be/...)"
                )
        return self

    @property
    def label(self) -> str:
        return self.name or self.id

    def to_video(self, base: VideoConfig) -> VideoConfig:
        """The full ``VideoConfig`` for this camera, on top of the shared one."""
        data = base.model_dump()
        data.update(
            source=self.source,
            source_id=self.id,
            device_id=self.device_id,
            rtsp_url=self.rtsp_url,
            youtube_url=self.youtube_url,
            path=self.path,
            loop=self.loop,
        )
        if self.source is VideoSourceType.FILE and data.get("playback_speed", 0) <= 0:
            # 0 means "as fast as possible" - right for a benchmark run from
            # the command line, wrong for a camera someone is watching: the
            # video would race by and a short clip would end in seconds.
            data["playback_speed"] = 1.0
        return VideoConfig(**data)


class ProcessingConfig(BaseModel):
    """Frame preprocessing and output settings owned by the Vision Core."""

    model_config = ConfigDict(extra="forbid")

    target_width: int | None = Field(
        default=960,
        description=(
            "Resize every captured frame to this width, preserving aspect ratio. "
            "Set to null to process at native capture resolution. The whole "
            "pipeline (inference and display) uses this single resolution, so "
            "bounding boxes never need rescaling."
        ),
    )
    show_window: bool = True
    window_name: str = "VisionGuard AI"
    show_hud: bool = True
    show_boxes: bool = True
    magic_effect: bool = Field(
        default=False,
        description=(
            "Draw a glowing trail and sparkles that trace how each tracked person "
            "moves (needs person_tracking). Nothing is drawn over people or their "
            "face/person boxes. Cosmetic only; can also be switched on and off "
            "live from the dashboard."
        ),
    )
    hud_max_detections: int = Field(
        default=5,
        description="Cap on per-detection HUD lines, so a crowd cannot fill the panel.",
    )
    mirror: bool = Field(
        default=False,
        description="Horizontally flip the frame. Useful for front-facing webcams.",
    )
    max_fps: float = Field(
        default=0.0,
        description="Throttle the loop to this FPS. 0 disables throttling.",
    )
    stall_timeout_sec: float = Field(
        default=30.0,
        description=(
            "Stop the pipeline when no frame has arrived for this long. Guards "
            "against a silently dead network stream. 0 disables the check."
        ),
    )
    stats_window: int = 30
    log_stats_every: int = Field(
        default=60,
        description="Headless mode logs a stats line every N frames. 0 disables.",
    )

    @field_validator("target_width")
    @classmethod
    def _validate_target_width(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("processing.target_width must be positive or null")
        return value

    @field_validator("hud_max_detections", "max_fps", "stall_timeout_sec", "log_stats_every")
    @classmethod
    def _validate_processing_non_negative(cls, value, info):
        if value < 0:
            raise ValueError(f"processing.{info.field_name} must be >= 0")
        return value

    @field_validator("stats_window")
    @classmethod
    def _validate_stats_window(cls, value: int) -> int:
        if value < 2:
            raise ValueError("processing.stats_window must be >= 2")
        return value


class AlertsConfig(BaseModel):
    """Which events become dashboard alerts, and how repeats are folded."""

    model_config = ConfigDict(extra="forbid")

    min_severity: Literal["info", "warning", "critical"] = Field(
        default="warning",
        description="Events at or above this severity become alerts.",
    )
    dedupe_seconds: float = Field(
        default=30.0,
        ge=0,
        description=(
            "The same event about the same subject on the same camera within "
            "this many seconds of the last one updates the open alert instead "
            "of opening a new one. 0 = every event is its own alert."
        ),
    )
    max_alerts: int = Field(
        default=500, ge=1, le=10000, description="Alerts kept in memory; oldest acknowledged go first."
    )


class DashboardConfig(BaseModel):
    """Read-only web view of a running pipeline."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: str = Field(
        default="127.0.0.1",
        description=(
            "Bind address. Defaults to localhost only. Binding to a network "
            "address (e.g. 0.0.0.0) always requires an access token: the one "
            "set in auth_token, or a random one generated and printed at startup."
        ),
    )
    auth_token: SecretStr | None = Field(
        default=None,
        description=(
            "Access token for the dashboard. Set it through the environment "
            "(VISIONGUARD_DASHBOARD_TOKEN / DASHBOARD_TOKEN), never in a "
            "committed config file. Optional on localhost; when unset on a "
            "network bind, a random per-run token is generated."
        ),
    )
    allowed_hosts: List[str] = Field(
        default_factory=list,
        description=(
            "Extra Host header values accepted (e.g. a LAN IP or hostname). "
            "localhost/127.0.0.1/::1 and a specific bind address are always "
            "accepted. Guards against DNS rebinding."
        ),
    )
    max_streams: int = Field(
        default=4,
        description="Concurrent live-video (MJPEG) viewers allowed; extra ones get 503.",
    )
    port: int = 8080
    stream_fps: float = Field(
        default=15.0,
        description=(
            "Cap on JPEG encoding for the live view. Analysis still runs at "
            "full rate; only the picture is throttled. 0 = every frame."
        ),
    )
    jpeg_quality: int = 75
    live_view: bool = Field(
        default=True,
        description=(
            "Show the camera at stream_fps even when analysis is slower, "
            "drawing the latest results over live frames. Applies to sources "
            "that capture on their own thread (RTSP); others show analysed "
            "frames. False = always show analysed frames only."
        ),
    )
    live_view_sync: bool = Field(
        default=True,
        description=(
            "Hold the live picture back by about one analysis interval (at "
            "most 3 s) so boxes are drawn where things were on the frame "
            "shown. False = always the newest frame, with boxes moved along "
            "their velocity: no delay, looser boxes."
        ),
    )
    max_events: int = Field(
        default=200, description="How many recent events the feed keeps."
    )
    max_cameras: int = Field(
        default=8,
        description=(
            "Most cameras one run may drive (1-32). Each camera is a full "
            "pipeline, so the real limit is your CPU, not this number."
        ),
    )
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    allow_settings_edit: bool = Field(
        default=True,
        description=(
            "Let the dashboard change modules' scalar settings (thresholds, "
            "timers, severity) via POST /api/settings. False = read-only."
        ),
    )
    allow_camera_edit: bool = Field(
        default=True,
        description=(
            "Let the dashboard change the camera list (POST /api/cameras). Set "
            "false to freeze the cameras to configs/cameras.yaml / video:."
        ),
    )
    allow_face_edit: bool = Field(
        default=True,
        description=(
            "Let the dashboard add, edit and delete registered users and "
            "enrol their faces (/api/face/*). False = the registry is read-only."
        ),
    )
    process_per_camera: bool = Field(
        default=False,
        description=(
            "Run each camera in its own process when there are several, so "
            "they use all CPU cores instead of sharing one Python interpreter. "
            "Faster with many cameras, but every camera then loads its own "
            "models (about 1 GB of RAM each)."
        ),
    )
    snapshot_dir: str = Field(
        default="data/snapshots",
        description="Where unknown-person snapshots are saved and served from (/api/snapshots).",
    )
    event_db: str = Field(
        default="data/events.db",
        description=(
            "SQLite file every event is appended to (entries/exits, unknown "
            "people, alerts...). Queried by the Reports page and exported as CSV."
        ),
    )
    face_db: str = Field(
        default="data/faces/faces.db",
        description=(
            "SQLite file holding registered users and their face templates "
            "(embeddings only, no photographs). Treat it as biometric data."
        ),
    )

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("dashboard.port must be between 1 and 65535")
        return value

    @field_validator("jpeg_quality")
    @classmethod
    def _validate_quality(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("dashboard.jpeg_quality must be between 1 and 100")
        return value

    @field_validator("stream_fps")
    @classmethod
    def _validate_stream_fps(cls, value: float) -> float:
        if value < 0:
            raise ValueError("dashboard.stream_fps must be >= 0 (0 = every frame)")
        return value

    @field_validator("max_events", "max_streams")
    @classmethod
    def _validate_positive_counts(cls, value: int, info) -> int:
        if value < 1:
            raise ValueError(f"dashboard.{info.field_name} must be >= 1")
        return value

    @field_validator("max_cameras")
    @classmethod
    def _validate_max_cameras(cls, value: int) -> int:
        if not 1 <= value <= 32:
            raise ValueError("dashboard.max_cameras must be between 1 and 32")
        return value

    @field_validator("auth_token", mode="before")
    @classmethod
    def _blank_token_is_none(cls, value):
        # An empty DASHBOARD_TOKEN= line must not become a valid empty token.
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator("auth_token")
    @classmethod
    def _validate_token_strength(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 16:
            raise ValueError("dashboard.auth_token must be at least 16 characters")
        return value

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value):
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


class ModuleConfig(BaseModel):
    """Per-module configuration.

    ``enabled`` is the only reserved key. Every other key is passed through to
    the module as ``params`` — the core never interprets them.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = False

    @property
    def params(self) -> Dict[str, Any]:
        """Module-specific settings (everything except ``enabled``)."""
        return dict(self.__pydantic_extra__ or {})


class VisionGuardConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    app: AppConfig = Field(default_factory=AppConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    modules: Dict[str, ModuleConfig] = Field(default_factory=dict)

    def enabled_module_names(self) -> List[str]:
        """Names of enabled modules, in the order they appear in the config."""
        return [name for name, cfg in self.modules.items() if cfg.enabled]

    def module_config(self, name: str) -> ModuleConfig:
        return self.modules.get(name, ModuleConfig())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_dotenv_if_available() -> None:
    """Load a local ``.env`` file when python-dotenv is installed.

    The dependency is optional on purpose: the platform must run in containers
    where configuration arrives purely through real environment variables.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _env(name: str) -> str | None:
    """Read ``VISIONGUARD_<NAME>`` falling back to bare ``<NAME>``."""
    value = os.getenv(f"{ENV_PREFIX}{name}")
    if value is None:
        value = os.getenv(name)
    return value.strip() if value is not None else None


def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _set_nested(data: Dict[str, Any], section: str, key: str, value: Any) -> None:
    data.setdefault(section, {})
    if not isinstance(data[section], dict):
        raise ConfigError(f"Config section '{section}' must be a mapping")
    data[section][key] = value


# Environment variable -> (config section, key, parser)
_ENV_OVERRIDES: Mapping[str, tuple[str, str, Any]] = {
    "VIDEO_SOURCE": ("video", "source", str),
    "VIDEO_SOURCE_ID": ("video", "source_id", str),
    "VIDEO_BACKEND": ("video", "backend", str),
    "VIDEO_DEVICE_ID": ("video", "device_id", int),
    "VIDEO_WIDTH": ("video", "width", int),
    "VIDEO_HEIGHT": ("video", "height", int),
    "VIDEO_FPS": ("video", "fps", int),
    "VIDEO_PATH": ("video", "path", str),
    "VIDEO_LOOP": ("video", "loop", _as_bool),
    "RTSP_URL": ("video", "rtsp_url", str),
    "RTSP_TRANSPORT": ("video", "rtsp_transport", str),
    "LOG_LEVEL": ("app", "log_level", str),
    "LOG_FILE": ("app", "log_file", str),
    "SHOW_WINDOW": ("processing", "show_window", _as_bool),
    "TARGET_WIDTH": ("processing", "target_width", int),
    "MIRROR": ("processing", "mirror", _as_bool),
    "MAX_FPS": ("processing", "max_fps", float),
    "DASHBOARD": ("dashboard", "enabled", _as_bool),
    "DASHBOARD_HOST": ("dashboard", "host", str),
    "DASHBOARD_PORT": ("dashboard", "port", int),
    "DASHBOARD_TOKEN": ("dashboard", "auth_token", str),
    "DASHBOARD_ALLOWED_HOSTS": ("dashboard", "allowed_hosts", str),
}


def apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay environment variables onto a raw config mapping."""
    for env_name, (section, key, parser) in _ENV_OVERRIDES.items():
        raw = _env(env_name)
        if raw is None or raw == "":
            continue
        try:
            _set_nested(data, section, key, parser(raw))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Environment variable {env_name}={raw!r} is not valid: {exc}"
            ) from exc
    return data


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    """Decide which YAML file to load."""
    if explicit:
        return Path(explicit)
    from_env = _env("CONFIG") or _env("CONFIG_PATH")
    if from_env:
        return Path(from_env)
    return DEFAULT_CONFIG_PATH


def load_raw_config(path: str | Path) -> Dict[str, Any]:
    """Read and parse a YAML config file into a plain mapping."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}\n"
            f"Create it or pass --config <path>."
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return data


def build_config(data: Mapping[str, Any]) -> VisionGuardConfig:
    """Validate a raw mapping into a :class:`VisionGuardConfig`."""
    try:
        return VisionGuardConfig.model_validate(data)
    except ValidationError as exc:
        details = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration:\n{details}") from exc


def load_config(
    path: str | Path | None = None,
    *,
    use_env: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> VisionGuardConfig:
    """Load, overlay and validate the full VisionGuard configuration.

    Args:
        path: Explicit YAML path. Falls back to ``VISIONGUARD_CONFIG`` then
            ``configs/config.yaml``.
        use_env: Apply ``.env`` / environment overrides.
        overrides: Final nested overrides, e.g. ``{"video": {"device_id": 1}}``.
            Used by ``backend.main`` for CLI flags.

    Raises:
        ConfigError: on a missing file, bad YAML or failed validation.
    """
    if use_env:
        _load_dotenv_if_available()

    data = load_raw_config(resolve_config_path(path))

    if use_env:
        data = apply_env_overrides(data)

    if overrides:
        for section, values in overrides.items():
            if not isinstance(values, Mapping):
                raise ConfigError(f"Override for '{section}' must be a mapping")
            for key, value in values.items():
                if value is None:
                    continue
                _set_nested(data, section, key, value)

    return build_config(data)
