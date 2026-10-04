import { getJson, postJson } from './client';
import {
  isCamerasResponse,
  isEventsResponse,
  isSetCamerasResponse,
  isSetModulesResponse,
  isStatusResponse,
} from './guards';
import type {
  CameraInput,
  CamerasResponse,
  EventsResponse,
  SetCamerasResponse,
  SetModulesResponse,
  StatusResponse,
} from './types';

/** Relative paths: the Vite proxy (dev) or a reverse proxy (prod) forwards them. */
export const ENDPOINTS = {
  status: '/api/status',
  events: '/api/events',
  stream: '/stream.mjpg',
  modules: '/api/modules',
  cameras: '/api/cameras',
  snapshot: '/api/snapshot.jpg',
} as const;

/** `?camera=<id>` selects a camera; without it the backend answers for its first. */
function withCamera(params: URLSearchParams, camera: string | null | undefined): URLSearchParams {
  if (camera) params.set('camera', camera);
  return params;
}

export function getStatus(signal?: AbortSignal, camera?: string | null): Promise<StatusResponse> {
  const query = withCamera(new URLSearchParams(), camera).toString();
  return getJson(query ? `${ENDPOINTS.status}?${query}` : ENDPOINTS.status, isStatusResponse, {
    signal,
  });
}

/** Events with id > `since`, newest first. The backend rejects limit > 200 with 422. */
export function getEvents(
  since: number,
  limit: number,
  signal?: AbortSignal,
  camera?: string | null,
): Promise<EventsResponse> {
  const query = withCamera(
    new URLSearchParams({ since: String(since), limit: String(limit) }),
    camera,
  );
  return getJson(`${ENDPOINTS.events}?${query}`, isEventsResponse, { signal });
}

/**
 * The backend ignores `r`; a new `nonce` gives the <img> a new URL, which
 * forces a fresh connection instead of a cached image.
 */
export function streamUrl(nonce: number, camera?: string | null): string {
  return `${ENDPOINTS.stream}?${withCamera(new URLSearchParams({ r: String(nonce) }), camera)}`;
}

/** One still frame, for thumbnails. Short-lived requests, unlike the MJPEG stream. */
export function snapshotUrl(camera: string, nonce: number): string {
  return `${ENDPOINTS.snapshot}?${new URLSearchParams({ camera, r: String(nonce) })}`;
}

export function getCameras(signal?: AbortSignal): Promise<CamerasResponse> {
  return getJson(ENDPOINTS.cameras, isCamerasResponse, { signal });
}

/**
 * Replace the camera list. `cameras` must be the complete desired list -
 * anything left out stops. Like setModules, the backend only queues the
 * change; the cameras restart in the background and can still fail to open.
 * Poll getCameras to see how each one comes up.
 */
export function setCameras(
  cameras: CameraInput[],
  signal?: AbortSignal,
): Promise<SetCamerasResponse> {
  return postJson(ENDPOINTS.cameras, { cameras }, isSetCamerasResponse, { signal });
}

/**
 * Replace the enabled module set. `enabled` must be the complete desired set
 * (dependencies included) - anything left out is turned off. The backend
 * only queues the change; it can still take a few seconds, or fail (e.g. a
 * missing dependency or model file), after this call has already resolved.
 */
export function setModules(enabled: string[], signal?: AbortSignal): Promise<SetModulesResponse> {
  return postJson(ENDPOINTS.modules, { enabled }, isSetModulesResponse, { signal });
}
