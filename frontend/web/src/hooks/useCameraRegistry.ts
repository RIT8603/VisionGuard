import type { ApiError } from '../api/client';
import type { CameraSummary } from '../api/cameras';

/**
 * What the portal knows about the camera registry.
 *
 * - unsupported: the backend has no camera discovery (today's state)
 * - loading:     the first registry request has not settled
 * - error:       the registry request failed
 * - ready:       the backend's camera list, possibly empty
 */
export type CameraRegistry =
  | { state: 'unsupported' }
  | { state: 'loading' }
  | { state: 'error'; error: ApiError }
  | { state: 'ready'; cameras: CameraSummary[] };

const UNSUPPORTED: CameraRegistry = { state: 'unsupported' };

/**
 * The single place the portal learns which cameras exist.
 *
 * PENDING BACKEND INTEGRATION: GET /api/cameras does not exist yet
 * (CAMERA_API_AVAILABLE in api/cameras.ts is false), so this makes no request
 * and reports "unsupported". When the backend adds it, poll
 * CAMERA_ENDPOINTS.list here (getJson + a shape guard, like usePipelineStatus)
 * and every camera page picks the data up without further changes.
 */
export function useCameraRegistry(): CameraRegistry {
  // Replace this body with the real request when CAMERA_API_AVAILABLE flips.
  return UNSUPPORTED;
}

/** The registry entry for `cameraId`, or null when unknown or unavailable. */
export function findCamera(registry: CameraRegistry, cameraId: string): CameraSummary | null {
  if (registry.state !== 'ready') return null;
  return registry.cameras.find((camera) => camera.camera_id === cameraId) ?? null;
}
