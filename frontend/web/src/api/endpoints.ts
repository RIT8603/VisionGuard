import { getJson, postJson } from './client';
import {
  isSetSettingsResponse,
  isSettingsResponse,
  isAckAllResponse,
  isAlertInfo,
  isAlertsResponse,
  isCamerasResponse,
  isEventsResponse,
  isSetCamerasResponse,
  isSetModulesResponse,
  isStatusResponse,
  isUploadedVideo,
  isVideosResponse,
} from './guards';
import type {
  EventLogPage,
  LiveReport,
  SetSettingsResponse,
  SettingsResponse,
  AckAllResponse,
  AlertInfo,
  AlertsResponse,
  AlertStatus,
  CameraInput,
  CamerasResponse,
  EventsResponse,
  SetCamerasResponse,
  SetModulesResponse,
  StatusResponse,
  UploadedVideo,
  VideosResponse,
} from './types';

/** Relative paths: the Vite proxy (dev) or a reverse proxy (prod) forwards them. */
export const ENDPOINTS = {
  status: '/api/status',
  events: '/api/events',
  stream: '/stream.mjpg',
  modules: '/api/modules',
  cameras: '/api/cameras',
  snapshot: '/api/snapshot.jpg',
  alerts: '/api/alerts',
  settings: '/api/settings',
  videos: '/api/videos',
  face: '/api/face',
  view: '/api/view',
  reports: '/api/reports',
} as const;

/** `?camera=<id>` selects a camera; without it the backend answers for its first. */
function withCamera(params: URLSearchParams, camera: string | null | undefined): URLSearchParams {
  if (camera) params.set('camera', camera);
  return params;
}

export function getStatus(
  signal?: AbortSignal,
  camera?: string | null,
  timeoutMs?: number,
): Promise<StatusResponse> {
  const query = withCamera(new URLSearchParams(), camera).toString();
  return getJson(query ? `${ENDPOINTS.status}?${query}` : ENDPOINTS.status, isStatusResponse, {
    signal,
    timeoutMs,
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
 * The backend ignores `r` and `t`; appending `t=${Date.now()}` ensures the browser
 * never serves a stale / cached MJPEG stream connection when switching cameras.
 */
export function streamUrl(nonce: number, camera?: string | null, client?: string): string {
  const params = new URLSearchParams({ r: String(nonce), t: String(Date.now()) });
  if (client) {
    params.set('client', client);
  }
  return `${ENDPOINTS.stream}?${withCamera(params, camera)}`;
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

/** Alerts across every camera, most recently updated first. */
export function getAlerts(
  status: AlertStatus | 'all',
  signal?: AbortSignal,
  camera?: string | null,
): Promise<AlertsResponse> {
  const query = withCamera(new URLSearchParams({ status, limit: '200' }), camera);
  return getJson(`${ENDPOINTS.alerts}?${query}`, isAlertsResponse, { signal });
}

/** Mark one alert as seen. Changes nothing else. */
export function acknowledgeAlert(id: number, signal?: AbortSignal): Promise<AlertInfo> {
  return postJson(`${ENDPOINTS.alerts}/${id}/ack`, {}, isAlertInfo, { signal });
}

/** Acknowledge every open alert, or only one camera's. */
export function acknowledgeAllAlerts(
  camera?: string | null,
  signal?: AbortSignal,
): Promise<AckAllResponse> {
  const query = withCamera(new URLSearchParams(), camera).toString();
  return postJson(query ? `${ENDPOINTS.alerts}/ack?${query}` : `${ENDPOINTS.alerts}/ack`, {}, isAckAllResponse, {
    signal,
  });
}

export function getSettings(signal?: AbortSignal): Promise<SettingsResponse> {
  return getJson(ENDPOINTS.settings, isSettingsResponse, { signal });
}

/**
 * Change some of one module's settings. Like setModules, the backend only
 * queues it: the modules restart, and a value the module rejects leaves the
 * previous settings running with `apply_error` set. Poll getSettings.
 */
export function setSettings(
  module: string,
  settings: Record<string, boolean | number | string>,
  signal?: AbortSignal,
): Promise<SetSettingsResponse> {
  return postJson(ENDPOINTS.settings, { module, settings }, isSetSettingsResponse, { signal });
}

export function getVideos(signal?: AbortSignal): Promise<VideosResponse> {
  return getJson(ENDPOINTS.videos, isVideosResponse, { signal });
}

/**
 * Send a video file to the machine running VisionGuard. Resolves with where
 * it was saved, which is what a file camera's path should be set to.
 * XMLHttpRequest rather than fetch: fetch cannot report upload progress, and
 * a CCTV recording can take minutes to send.
 */
export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<UploadedVideo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${ENDPOINTS.videos}?name=${encodeURIComponent(file.name)}`);
    xhr.withCredentials = true;
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* handled below */
      }
      if (xhr.status >= 200 && xhr.status < 300 && isUploadedVideo(body)) {
        resolve(body);
        return;
      }
      const detail =
        typeof body === 'object' && body !== null && typeof (body as { error?: unknown }).error === 'string'
          ? (body as { error: string }).error
          : `Upload failed (HTTP ${xhr.status})`;
      reject(new Error(detail));
    };
    xhr.onerror = () => reject(new Error('Could not reach VisionGuard to upload the video'));
    xhr.onabort = () => reject(new Error('Upload cancelled'));
    signal?.addEventListener('abort', () => xhr.abort(), { once: true });
    xhr.send(file);
  });
}

// ---------- view options ----------

export interface ViewOptions {
  /** Glowing trail and sparkles tracing how tracked people move. */
  magic_effect: boolean;
}

const isViewOptions = (value: unknown): value is ViewOptions =>
  typeof value === 'object' && value !== null && typeof (value as ViewOptions).magic_effect === 'boolean';

export function getViewOptions(signal?: AbortSignal): Promise<ViewOptions> {
  return getJson(ENDPOINTS.view, isViewOptions, { signal });
}

export function setMagicEffect(on: boolean, signal?: AbortSignal): Promise<ViewOptions> {
  return postJson(ENDPOINTS.view, { magic_effect: on }, isViewOptions, { signal });
}

// ---------- reports ----------

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

export function getLiveReport(signal?: AbortSignal): Promise<LiveReport> {
  return getJson(
    `${ENDPOINTS.reports}/live`,
    (v): v is LiveReport => isObj(v) && Array.isArray(v.cameras),
    { signal },
  );
}

export interface EventLogFilters {
  camera?: string;
  event?: string;
  identity?: string;
  day?: string;
  limit?: number;
  offset?: number;
}

function reportQuery(filters: EventLogFilters): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== '' && value !== null) params.set(key, String(value));
  }
  const text = params.toString();
  return text ? `?${text}` : '';
}

export function getEventLog(
  filters: EventLogFilters,
  signal?: AbortSignal,
): Promise<EventLogPage> {
  return getJson(
    `${ENDPOINTS.reports}/events${reportQuery(filters)}`,
    (v): v is EventLogPage => isObj(v) && typeof v.total === 'number' && Array.isArray(v.events),
    { signal },
  );
}

/** A link that downloads the filtered event log as CSV. */
export function eventLogCsvUrl(filters: EventLogFilters): string {
  const { limit: _limit, offset: _offset, ...rest } = filters;
  return `${ENDPOINTS.reports}/events.csv${reportQuery(rest)}`;
}

/** A link that downloads a day's attendance (first entry, last exit) as CSV. */
export function attendanceCsvUrl(day?: string): string {
  return `${ENDPOINTS.reports}/attendance.csv${reportQuery({ day })}`;
}

export function snapshotImageUrl(name: string): string {
  return `/api/snapshots/${encodeURIComponent(name)}`;
}
