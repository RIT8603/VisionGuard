import { hasFrame } from '../api/guards';
import type { StatusResponse } from '../api/types';
import { OFFLINE_AFTER_FAILURES, STALE_AFTER_MS } from '../config';

/**
 * What the dashboard should show, derived from the status poll.
 *
 * - loading:  no successful status response yet
 * - offline:  the last OFFLINE_AFTER_FAILURES polls failed
 * - stopped:  the backend reported running=false (it exits right after)
 * - waiting:  backend up, no frame processed yet (camera still opening)
 * - stale:    frame_id has not advanced for STALE_AFTER_MS
 * - live:     frames are flowing
 */
export type ViewState = 'loading' | 'waiting' | 'live' | 'stale' | 'stopped' | 'offline';

export const VIEW_STATE_LABEL: Record<ViewState, string> = {
  loading: 'Connecting',
  waiting: 'Waiting for video',
  live: 'Live',
  stale: 'Stale',
  stopped: 'Stopped',
  offline: 'Offline',
};

interface ViewStateInput {
  data: StatusResponse | null;
  failures: number;
  /** Milliseconds since frame_id last changed. */
  staleForMs: number;
}

export function deriveViewState({ data, failures, staleForMs }: ViewStateInput): ViewState {
  if (failures >= OFFLINE_AFTER_FAILURES) return 'offline';
  if (data === null) return 'loading';
  if (!data.running) return 'stopped';
  if (!hasFrame(data.frame)) return 'waiting';
  return staleForMs > STALE_AFTER_MS ? 'stale' : 'live';
}

/** States in which the backend is answering and the stream can be opened. */
export function isReachable(state: ViewState): boolean {
  return state === 'waiting' || state === 'live' || state === 'stale';
}
