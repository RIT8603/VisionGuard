/**
 * Runtime shape checks for API responses.
 *
 * Deliberately shallow: they verify what the UI branches on, so a malformed
 * or unexpected payload surfaces as an error instead of a render crash.
 */

import type {
  AddFaceSampleResponse,
  FaceReading,
  FaceStatus,
  FaceUser,
  FaceUsersResponse,
  SetSettingsResponse,
  SettingsResponse,
  AckAllResponse,
  AlertInfo,
  AlertsResponse,
  CameraInfo,
  CamerasResponse,
  EventsResponse,
  FeedEvent,
  FrameSummary,
  PipelineStats,
  SetCamerasResponse,
  UploadedVideo,
  VideosResponse,
  SetModulesResponse,
  StatusResponse,
} from './types';

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function isStatusResponse(value: unknown): value is StatusResponse {
  if (!isObject(value)) return false;
  const { info } = value;
  return (
    typeof value.running === 'boolean' &&
    typeof value.uptime_sec === 'number' &&
    typeof value.event_count === 'number' &&
    isObject(info) &&
    Array.isArray(info.modules) &&
    isObject(value.stats) &&
    isObject(value.frame)
  );
}

function isFeedEvent(value: unknown): value is FeedEvent {
  return (
    isObject(value) &&
    typeof value.id === 'number' &&
    typeof value.event === 'string' &&
    typeof value.timestamp === 'string'
  );
}

export function isEventsResponse(value: unknown): value is EventsResponse {
  return isObject(value) && Array.isArray(value.events) && value.events.every(isFeedEvent);
}

export function isSetModulesResponse(value: unknown): value is SetModulesResponse {
  return (
    isObject(value) &&
    typeof value.status === 'string' &&
    Array.isArray(value.enabled) &&
    value.enabled.every((entry) => typeof entry === 'string')
  );
}

function isCameraInfo(value: unknown): value is CameraInfo {
  return (
    isObject(value) &&
    typeof value.id === 'string' &&
    typeof value.name === 'string' &&
    typeof value.source === 'string' &&
    typeof value.status === 'string'
  );
}

export function isCamerasResponse(value: unknown): value is CamerasResponse {
  return (
    isObject(value) &&
    Array.isArray(value.cameras) &&
    value.cameras.every(isCameraInfo) &&
    typeof value.max_cameras === 'number' &&
    typeof value.editable === 'boolean'
  );
}

export function isSetCamerasResponse(value: unknown): value is SetCamerasResponse {
  return isObject(value) && typeof value.status === 'string' && Array.isArray(value.cameras);
}

/** False before the first frame, when the backend sends `frame: {}`. */
export function hasFrame(frame: StatusResponse['frame']): frame is FrameSummary {
  const candidate: unknown = frame;
  return (
    isObject(candidate) &&
    typeof candidate.frame_id === 'number' &&
    isObject(candidate.resolution) &&
    isObject(candidate.detections) &&
    isObject(candidate.tracks)
  );
}

/** False before the first frame, when the backend sends `stats: {}`. */
export function hasStats(stats: StatusResponse['stats']): stats is PipelineStats {
  const candidate: unknown = stats;
  return (
    isObject(candidate) &&
    typeof candidate.fps === 'number' &&
    typeof candidate.frames_processed === 'number'
  );
}

export function isAlertInfo(value: unknown): value is AlertInfo {
  return (
    isObject(value) &&
    typeof value.id === 'number' &&
    typeof value.camera === 'string' &&
    typeof value.event === 'string' &&
    typeof value.status === 'string' &&
    typeof value.count === 'number' &&
    typeof value.last_seen === 'string'
  );
}

export function isAlertsResponse(value: unknown): value is AlertsResponse {
  return (
    isObject(value) &&
    Array.isArray(value.alerts) &&
    value.alerts.every(isAlertInfo) &&
    typeof value.open_count === 'number'
  );
}

export function isAckAllResponse(value: unknown): value is AckAllResponse {
  return isObject(value) && typeof value.acknowledged === 'number';
}

export function isSettingsResponse(value: unknown): value is SettingsResponse {
  return (
    isObject(value) &&
    typeof value.editable === 'boolean' &&
    isObject(value.modules) &&
    Object.values(value.modules).every(
      (entry) => isObject(entry) && typeof entry.enabled === 'boolean' && isObject(entry.settings),
    )
  );
}

export function isSetSettingsResponse(value: unknown): value is SetSettingsResponse {
  return isObject(value) && typeof value.status === 'string' && typeof value.module === 'string';
}

export function isUploadedVideo(value: unknown): value is UploadedVideo {
  return (
    isObject(value) &&
    typeof value.name === 'string' &&
    typeof value.path === 'string' &&
    typeof value.size === 'number'
  );
}

export function isVideosResponse(value: unknown): value is VideosResponse {
  return isObject(value) && Array.isArray(value.videos) && value.videos.every(isUploadedVideo);
}

// ---------- face registry ----------

export function isFaceStatus(value: unknown): value is FaceStatus {
  return (
    isObject(value) &&
    typeof value.editable === 'boolean' &&
    isObject(value.models) &&
    typeof value.models.ready === 'boolean' &&
    Array.isArray(value.poses) &&
    typeof value.samples_per_pose === 'number' &&
    typeof value.users === 'number'
  );
}

export function isFaceUser(value: unknown): value is FaceUser {
  return (
    isObject(value) &&
    typeof value.id === 'number' &&
    typeof value.name === 'string' &&
    typeof value.active === 'boolean' &&
    typeof value.enrolled === 'boolean' &&
    isObject(value.poses) &&
    Array.isArray(value.missing_poses)
  );
}

export function isFaceUsersResponse(value: unknown): value is FaceUsersResponse {
  return isObject(value) && Array.isArray(value.users) && value.users.every(isFaceUser);
}

export function isFaceReading(value: unknown): value is FaceReading {
  return (
    isObject(value) &&
    typeof value.ok === 'boolean' &&
    typeof value.faces === 'number' &&
    Array.isArray(value.problems) &&
    (value.face === null || isObject(value.face))
  );
}

export function isAddFaceSampleResponse(value: unknown): value is AddFaceSampleResponse {
  return isObject(value) && typeof value.pose === 'string' && isFaceUser(value.user);
}

export function isDeleted(value: unknown): value is { deleted: number } {
  return isObject(value) && typeof value.deleted === 'number';
}
