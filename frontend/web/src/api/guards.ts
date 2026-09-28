/**
 * Runtime shape checks for API responses.
 *
 * Deliberately shallow: they verify what the UI branches on, so a malformed
 * or unexpected payload surfaces as an error instead of a render crash.
 */

import type {
  EventsResponse,
  FeedEvent,
  FrameSummary,
  PipelineStats,
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
