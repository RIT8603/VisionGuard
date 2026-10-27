/**
 * Response shapes of the VisionGuard dashboard API.
 *
 * The backend returns plain dicts with no schema, so these are maintained by
 * hand. Sources of truth:
 *   backend/services/state.py         StatusResponse, event ids
 *   computer_vision/core/context.py   FrameSummary, Detection, Track, Pose, events
 *   computer_vision/utils/timing.py   PipelineStats
 */

export type Severity = 'info' | 'warning' | 'critical';
export type ModuleType = 'detection' | 'recognition' | 'tracking' | 'analysis';
export type SourceType = 'webcam' | 'file' | 'rtsp' | 'youtube';

/** [x1, y1, x2, y2] in pixels of `FrameSummary.resolution`. */
export type BBox = [number, number, number, number];

export interface ModuleInfo {
  name: string;
  type: ModuleType;
  version: string;
}

export interface RunInfo {
  app: string;
  version: string;
  source_type: SourceType;
  /** "webcam-0", "file-<stem>", "rtsp-0" or a configured id. */
  source_id: string;
  /** Which camera this status describes (the camera id; equals source_id). */
  camera_id?: string;
  /** The camera's display name. */
  camera_name?: string;
  /** Display text such as "960px wide" or "native" - not dimensions. */
  resolution: string;
  /** Captured once at startup; a module disabled mid-run stays listed. */
  modules: ModuleInfo[];
}

export interface PipelineStats {
  frames_read: number;
  frames_processed: number;
  frames_failed: number;
  /** Seconds since the source opened (not the same as StatusResponse.uptime_sec). */
  uptime_sec: number;
  /** Rolling over the last `processing.stats_window` frames. */
  fps: number;
  /** Whole-run average. */
  overall_fps: number;
  latency_ms: number;
  capture_ms: number;
  module_latency_ms: Record<string, number>;
}

export interface Detection {
  class: string;
  confidence: number;
  bbox: BBox;
  class_id?: number;
  /** face_recognition: { identity: string | null, similarity: number, recognised: boolean } */
  extra?: Record<string, unknown>;
}

export interface Track {
  track_id: number;
  class: string;
  bbox: BBox;
  confidence: number;
  age: number;
  hits: number;
  extra?: Record<string, unknown>;
}

/** [x, y, confidence] */
export type Keypoint = [number, number, number];

export interface Pose {
  bbox: BBox;
  confidence: number;
  /** COCO joint names; only joints the model was confident about are present. */
  keypoints: Partial<Record<string, Keypoint>>;
  extra?: Record<string, unknown>;
}

export type KnownEventType =
  | 'zone_entry'
  | 'zone_exit'
  | 'zone_dwell'
  | 'fall_detected'
  | 'fall_recovered'
  | 'attendance_check_in'
  | 'attendance_check_out'
  | 'unknown_person'
  | 'loitering'
  | 'intrusion'
  | 'tailgating'
  | 'line_crossed'
  | 'abandoned_object'
  | 'abandoned_object_removed'
  | 'inventory_low'
  | 'inventory_high'
  | 'inventory_restored'
  | 'low_light_started'
  | 'low_light_ended'
  | 'weapon_detected'
  | 'armed_person'
  | 'fire_detected'
  | 'smoke_detected'
  | 'ppe_violation'
  | 'plate_read'
  | 'person_reidentified'
  | 'crowd_formed'
  | 'crowd_cleared';

/** Keys observed in the current analysis modules; any module may add others. */
export interface EventSubject {
  track_id?: number;
  class?: string;
  identity?: string;
  count?: number;
  [key: string]: unknown;
}

export interface EventData {
  zone?: string;
  dwell_seconds?: number;
  frames_inside?: number;
  threshold_seconds?: number;
  score?: number;
  signals?: Record<string, number>;
  consecutive_frames?: number;
  sightings?: number;
  duration_seconds?: number;
  unseen_seconds?: number;
  closest_similarity?: number;
  consecutive_sightings?: number;
  reason?: string;
  [key: string]: unknown;
}

/** An event as it appears inside `FrameSummary.events` (no id). */
export interface FrameEvent {
  // `string & {}` keeps autocomplete for known types while allowing new ones.
  event: KnownEventType | (string & {});
  module: string;
  frame_id: number;
  /** ISO 8601 with UTC offset. */
  timestamp: string;
  severity: Severity;
  subject?: EventSubject;
  data?: EventData;
}

/** An event from /api/events. Ids increase by one and restart with the backend. */
export interface FeedEvent extends FrameEvent {
  id: number;
}

export interface FrameSummary {
  frame_id: number;
  source_id: string;
  timestamp: string;
  resolution: { width: number; height: number };
  /** Keyed by the producing module's name. */
  detections: Record<string, Detection[]>;
  tracks: Record<string, Track[]>;
  poses: Record<string, Pose[]>;
  /** This frame only. Use /api/events for the feed. */
  events: FrameEvent[];
  metadata: { latency_ms?: number; fps?: number; [key: string]: unknown };
  /** module name -> "ExceptionType: message" for modules that failed this frame. */
  errors: Record<string, string>;
}

export type EmptyObject = Record<string, never>;

export interface StatusResponse {
  /** Flips to false only moments before the backend process exits. */
  running: boolean;
  /** Wall-clock seconds since the dashboard started. */
  uptime_sec: number;
  info: RunInfo;
  /** `{}` until the first frame has been processed. */
  stats: PipelineStats | EmptyObject;
  /** `{}` until the first frame has been processed. */
  frame: FrameSummary | EmptyObject;
  /** Highest event id issued so far. */
  event_count: number;
}

export interface EventsResponse {
  /** Newest first. */
  events: FeedEvent[];
}

/** Response to POST /api/modules. Acceptance only - the pipeline restarts
 * in the background; poll /api/status to see when it has caught up. */
export interface SetModulesResponse {
  status: string;
  enabled: string[];
}

/** What a camera is doing right now, as reported by GET /api/cameras. */
export type CameraStatus = 'starting' | 'live' | 'stalled' | 'offline' | 'failed';

/** One camera as GET /api/cameras describes it. Credentials in `rtsp_url` are masked (`***`). */
export interface CameraInfo {
  id: string;
  name: string;
  source: SourceType;
  status: CameraStatus;
  /** Why the camera failed, when status is "failed". */
  error: string | null;
  fps: number | null;
  resolution: { width: number; height: number } | null;
  /** Highest event id issued by this camera so far. */
  event_count: number;
  device_id?: number;
  rtsp_url?: string | null;
  youtube_url?: string | null;
  path?: string | null;
  loop?: boolean;
  /** Backend module keys this camera runs on its own; null = the shared selection. */
  modules?: string[] | null;
}

export interface CamerasResponse {
  cameras: CameraInfo[];
  /** The most cameras one run may drive (dashboard.max_cameras). */
  max_cameras: number;
  /** False when the backend refuses camera changes. */
  editable: boolean;
  /** Why the last requested change could not be applied, if it could not. */
  apply_error: string | null;
  /** Backend module keys of the shared selection (the Modules page). */
  shared_modules?: string[] | null;
}

/** One camera in a POST /api/cameras body. Send only the field its source uses. */
export interface CameraInput {
  /** Keep the existing id to keep the camera's history; omit it for a new camera. */
  id?: string;
  name?: string;
  source: SourceType;
  rtsp_url?: string;
  youtube_url?: string;
  device_id?: number;
  path?: string;
  loop?: boolean;
  /** This camera's own backend module keys. Omit to follow the shared selection. */
  modules?: string[];
}

/** Response to POST /api/cameras. Acceptance only - cameras restart in the background. */
export interface SetCamerasResponse {
  status: string;
  cameras: CameraInfo[];
}

/** Open until someone acknowledges it on the Alerts page. */
export type AlertStatus = 'open' | 'acknowledged';

/**
 * One alert from GET /api/alerts: an event at or above the configured
 * severity, with repeats of the same thing on the same camera folded in.
 */
export interface AlertInfo {
  id: number;
  camera: string;
  event: KnownEventType | (string & {});
  module: string;
  severity: Severity;
  subject: EventSubject;
  data: EventData;
  /** ISO 8601 timestamps of the first and latest folded event. */
  first_seen: string;
  last_seen: string;
  /** How many events this alert stands for. */
  count: number;
  status: AlertStatus;
  acknowledged_at: string | null;
}

export interface AlertsResponse {
  /** Most recently updated first. */
  alerts: AlertInfo[];
  /** Across every camera and status filter. */
  open_count: number;
  open_critical: number;
  total: number;
  min_severity: Severity;
}

export interface AckAllResponse {
  acknowledged: number;
}

/** How a setting is edited: a checkbox, a number field, or a severity picker. */
export type SettingKind = 'bool' | 'int' | 'float' | 'severity';

export interface SettingValue {
  value: boolean | number | Severity;
  kind: SettingKind;
}

export interface ModuleSettings {
  enabled: boolean;
  settings: Record<string, SettingValue>;
}

/** GET /api/settings: each module's scalar settings the dashboard may change. */
export interface SettingsResponse {
  editable: boolean;
  /** Why the last change could not be applied; null when it was. */
  apply_error: string | null;
  modules: Record<string, ModuleSettings>;
}

export interface SetSettingsResponse {
  status: string;
  module: string;
  settings: Record<string, boolean | number | string>;
}

/** A video saved on the machine running VisionGuard (`/api/videos`). */
export interface UploadedVideo {
  name: string;
  /** Absolute path on that machine: what a file camera's `path` is set to. */
  path: string;
  size: number;
}

export interface VideosResponse {
  videos: UploadedVideo[];
}

// ---------- reports (/api/reports/*) ----------

export interface LiveCamera {
  id: string;
  name: string;
  status: CameraStatus;
  people_in_view: number;
  recognised: string[];
  unknown_faces: number;
  spoof_faces: number;
}

export interface PersonInside {
  identity: string;
  since: string;
  camera: string;
}

export interface LiveReport {
  cameras: LiveCamera[];
  people_in_view: number;
  today?: {
    entered: number;
    exited: number;
    inside: number;
    people_inside: PersonInside[];
    truncated: boolean;
    by_type: Record<string, number>;
  };
}

export interface LoggedEvent {
  id: number;
  camera: string;
  timestamp: string;
  event: string;
  module: string;
  severity: Severity;
  track_id: number | null;
  identity: string | null;
  subject: Record<string, unknown>;
  data: Record<string, unknown>;
}

export interface EventLogPage {
  total: number;
  events: LoggedEvent[];
}
