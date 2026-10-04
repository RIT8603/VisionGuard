/**
 * Camera-scoped API — PENDING BACKEND INTEGRATION.
 *
 * The backend still runs a single source with one status feed, one event
 * buffer and one MJPEG stream (see endpoints.ts). It has no camera registry
 * and no camera-scoped routes. This module describes the contract the
 * frontend expects once those exist, so pages can be built against it now:
 *
 *   GET /api/cameras
 *   GET /api/cameras/{cameraId}/status
 *   GET /api/cameras/{cameraId}/events
 *   GET /api/cameras/{cameraId}/snapshot.jpg
 *   GET /api/cameras/{cameraId}/stream.mjpg
 *
 * Nothing here issues a request. While CAMERA_API_AVAILABLE is false, no code
 * path may call these endpoints; useCameraRegistry() reports "unsupported".
 * The shapes below are an expectation, not a mirror of backend code: confirm
 * them against the real implementation before flipping the flag.
 */

import type { FeedEvent, SourceType, StatusResponse } from './types';

/** Flip to true only once the backend serves the routes above. */
export const CAMERA_API_AVAILABLE = false;

/** Expected camera states. `string & {}` tolerates states a backend adds later. */
export type CameraStatus = 'online' | 'starting' | 'stale' | 'offline' | 'stopped' | (string & {});

/** One entry of GET /api/cameras. */
export interface CameraSummary {
  /** Stable id assigned by the backend; used in every camera-scoped URL. */
  camera_id: string;
  /** Human-readable name. */
  name: string;
  source_type: SourceType | (string & {});
  status: CameraStatus;
  /** ISO 8601 time of the most recent processed frame; null before the first. */
  last_frame_at: string | null;
}

export interface CameraListResponse {
  cameras: CameraSummary[];
}

/**
 * GET /api/cameras/{cameraId}/status — expected to match today's /api/status
 * so the existing panels (StatsPanel, FrameResults, ModulesPanel, …) can
 * render it unchanged.
 */
export interface CameraStatusResponse extends StatusResponse {
  camera_id: string;
}

/** An event from GET /api/cameras/{cameraId}/events. Today's events carry no camera id. */
export interface CameraFeedEvent extends FeedEvent {
  camera_id: string;
}

export interface CameraEventsResponse {
  events: CameraFeedEvent[];
}

const segment = (cameraId: string) => encodeURIComponent(cameraId);

/** Path builders for the pending contract. Do not request them while the flag is false. */
export const CAMERA_ENDPOINTS = {
  list: '/api/cameras',
  status: (cameraId: string) => `/api/cameras/${segment(cameraId)}/status`,
  events: (cameraId: string) => `/api/cameras/${segment(cameraId)}/events`,
  snapshot: (cameraId: string) => `/api/cameras/${segment(cameraId)}/snapshot.jpg`,
  stream: (cameraId: string) => `/api/cameras/${segment(cameraId)}/stream.mjpg`,
} as const;

/** Same nonce trick as streamUrl(): a new URL forces a fresh connection. */
export function cameraStreamUrl(cameraId: string, nonce: number): string {
  return `${CAMERA_ENDPOINTS.stream(cameraId)}?r=${nonce}`;
}
