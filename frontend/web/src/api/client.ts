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

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * POST a JSON body and verify the response shape.
 *
 * The only endpoint that needs this today, `/api/modules`, answers a
 * rejected request with `{"error": "<reason>"}` rather than an empty body,
 * so a non-2xx response's message is taken from there when present.
 */
export async function postJson<T>(
  path: string,
  body: unknown,
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

  const fail = (error: unknown, kind: ApiErrorKind, message: string): never => {
    if (signal?.aborted) throw error;
    if (timedOut) throw new ApiError('timeout', `No response from ${path} within ${timeoutMs} ms`);
    throw new ApiError(kind, message);
  };

  try {
    let response: Response;
    try {
      response = await fetch(path, {
        method: 'POST',
        signal: controller.signal,
        cache: 'no-store',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (error) {
      return fail(error, 'network', `Could not reach ${path}`);
    }

    let parsed: unknown;
    try {
      parsed = await response.json();
    } catch (error) {
      if (!response.ok) {
        throw new ApiError('http', `${path} returned HTTP ${response.status}`, response.status);
      }
      return fail(error, 'parse', `${path} returned a body that is not JSON`);
    }

    if (!response.ok) {
      const detail = isPlainObject(parsed) && typeof parsed.error === 'string' ? parsed.error : undefined;
      throw new ApiError('http', detail ?? `${path} returned HTTP ${response.status}`, response.status);
    }

    if (!isValid(parsed)) {
      throw new ApiError('shape', `${path} returned an unexpected response shape`);
    }
    return parsed;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
