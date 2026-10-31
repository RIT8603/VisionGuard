/**
 * The wake service on the frontend server (deploy/wake/vgwake.py): starts the
 * GPU backend on demand and reports its real startup stages.
 *
 * Not part of the backend API: it answers while the backend is stopped. A
 * build served without it (e.g. the Vite dev server) gets `null` from
 * getWakeStatus, and the portal opens directly as before.
 */

export type WakeState = 'ready' | 'starting' | 'asleep' | 'stopping' | 'error' | 'unknown';
export type StageStatus = 'done' | 'active' | 'pending' | 'failed';

export interface WakeStage {
  id: string;
  label: string;
  status: StageStatus;
  /** Measured seconds this stage took (done) or has taken so far (active). */
  seconds?: number;
  /** Typical duration from previous real startups (see eta_basis). */
  typical_seconds: number;
}

/** Present only while AWS has refused the start with InsufficientInstanceCapacity. */
export interface CapacityWait {
  attempts_failed: number;
  max_attempts: number;
  /** Seconds until the next automatic StartInstances attempt. */
  next_retry_in: number;
}

export interface WakeStatus {
  state: WakeState;
  ready: boolean;
  message: string;
  error: string | null;
  /** Why `error` is set: "capacity" = AWS had no GPU after every retry. */
  error_kind?: 'capacity' | 'timeout' | 'aws' | null;
  capacity?: CapacityWait | null;
  /** Share of startup stages verified complete, weighted by their typical duration. */
  progress: number;
  eta_seconds: number | null;
  eta_basis: string;
  elapsed_seconds: number | null;
  stages: WakeStage[];
  idle_minutes: number;
}

const isWakeStatus = (value: unknown): value is WakeStatus =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as WakeStatus).state === 'string' &&
  typeof (value as WakeStatus).ready === 'boolean' &&
  Array.isArray((value as WakeStatus).stages);

async function call(method: 'GET' | 'POST', path: string): Promise<WakeStatus | null> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(path, {
      method,
      cache: 'no-store',
      credentials: 'same-origin',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    // 202 starting, 429 budget/rate limit and 503 no AWS access all carry a status body.
    const body: unknown = await response.json().catch(() => null);
    return isWakeStatus(body) ? body : null;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

/** Current backend state, or null when there is no wake service (or it is unreachable). */
export const getWakeStatus = () => call('GET', '/wake/status');

/** Ask for the backend to be started. Safe to repeat: the service ignores duplicates. */
export const requestWake = () => call('POST', '/wake/start');
