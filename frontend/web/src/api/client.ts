import { REQUEST_TIMEOUT_MS } from '../config';

/**
 * Why a request failed.
 *
 * Note that when the backend is down, the Vite dev proxy answers with a 5xx
 * (kind `http`), whereas a production build talking to it directly gets a
 * `network` error. Callers should treat both as "backend unreachable".
 */
export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'parse' | 'shape';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
  }
}

export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  return new ApiError('network', error instanceof Error ? error.message : String(error));
}

interface RequestOptions {
  /** Aborting this cancels the request; the rejection is then not an ApiError. */
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** GET a JSON endpoint and verify its shape. */
export async function getJson<T>(
  path: string,
  isValid: (value: unknown) => value is T,
  { signal, timeoutMs = REQUEST_TIMEOUT_MS }: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const cancel = () => controller.abort();
  if (signal?.aborted) controller.abort();
  signal?.addEventListener('abort', cancel, { once: true });

  // Caller cancellation passes through untouched so hooks can ignore it.
  const fail = (error: unknown, kind: ApiErrorKind, message: string): never => {
    if (signal?.aborted) throw error;
    if (timedOut) throw new ApiError('timeout', `No response from ${path} within ${timeoutMs} ms`);
    throw new ApiError(kind, message);
  };

  try {
    let response: Response;
    try {
      response = await fetch(path, {
        signal: controller.signal,
        cache: 'no-store',
        headers: { Accept: 'application/json' },
      });
    } catch (error) {
      return fail(error, 'network', `Could not reach ${path}`);
    }

    if (!response.ok) {
      throw new ApiError('http', `${path} returned HTTP ${response.status}`, response.status);
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch (error) {
      return fail(error, 'parse', `${path} returned a body that is not JSON`);
    }

    if (!isValid(body)) {
      throw new ApiError('shape', `${path} returned an unexpected response shape`);
    }
    return body;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
