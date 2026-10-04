/**
 * Runtime shape checks for API responses.
 *
 * Deliberately shallow: they verify what the UI branches on, so a malformed
 * or unexpected payload surfaces as an error instead of a render crash.
 */

import type {
  CameraInfo,
  CamerasResponse,
  EventsResponse,
  FeedEvent,
  FrameSummary,
  PipelineStats,
  SetCamerasResponse,
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
