import { getJson } from './client';
import { isEventsResponse, isStatusResponse } from './guards';
import type { EventsResponse, StatusResponse } from './types';

/** Relative paths: the Vite proxy (dev) or a reverse proxy (prod) forwards them. */
export const ENDPOINTS = {
  status: '/api/status',
  events: '/api/events',
  stream: '/stream.mjpg',
} as const;

export function getStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return getJson(ENDPOINTS.status, isStatusResponse, { signal });
}

/** Events with id > `since`, newest first. The backend rejects limit > 200 with 422. */
export function getEvents(
  since: number,
  limit: number,
  signal?: AbortSignal,
): Promise<EventsResponse> {
  const query = new URLSearchParams({ since: String(since), limit: String(limit) });
  return getJson(`${ENDPOINTS.events}?${query}`, isEventsResponse, { signal });
}

/**
 * The backend ignores the query string; a new `nonce` gives the <img> a new
 * URL, which forces a fresh connection instead of a cached image.
 */
export function streamUrl(nonce: number): string {
  return `${ENDPOINTS.stream}?r=${nonce}`;
}
