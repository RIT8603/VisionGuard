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
export type SourceType = 'webcam' | 'file' | 'rtsp';

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
  | 'unknown_person';

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
