/** Timing and size limits for talking to the backend. */

/** Matches the legacy dashboard (frontend/static/index.html). */
export const STATUS_POLL_MS = 500;

export const EVENTS_POLL_MS = 1000;

/**
 * Always request the maximum. /api/events returns the *newest* `limit` events
 * above the cursor, so a smaller limit silently drops the older part of a burst.
 */
export const EVENTS_LIMIT = 200;

/** The backend keeps 200 events by default (dashboard.max_events). */
export const MAX_FEED_EVENTS = 200;

export const REQUEST_TIMEOUT_MS = 5000;

/** Consecutive failed status polls before the backend counts as offline. */
export const OFFLINE_AFTER_FAILURES = 3;

/** Debounce delay before applying camera switch to avoid connection churn. */
export const CAMERA_SWITCH_DEBOUNCE_MS = 200;

/** No new frame_id for this long means the video has stalled. */
export const STALE_AFTER_MS = 3000;

/** `uptime_sec` dropping by more than this means the backend restarted. */
export const RESTART_TOLERANCE_SEC = 1;

/** No new stream frame for this long: the connection is replaced. */
export const STREAM_STALL_MS = 3000;

/** Longest wait between stream reconnect attempts (the first one is immediate). */
export const STREAM_MAX_BACKOFF_MS = 5000;

/** How often the live feed logs its delivery figures ([vg-stream] in the console). */
export const STREAM_LOG_MS = 5000;

/** How often the camera list (status of every camera) is refreshed. */
export const CAMERAS_POLL_MS = 1000;

/** How often a camera thumbnail is replaced. Each refresh is one short request. */
export const TILE_REFRESH_MS = 800;

/** How often the alert list (and the nav badge) is refreshed. */
export const ALERTS_POLL_MS = 2000;
