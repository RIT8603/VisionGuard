/**
 * Client for the help assistant's serverless endpoint (api/assistant.ts).
 * Separate from the backend client: different server, different failures.
 */

export interface AssistantTurn {
  role: 'user' | 'assistant';
  content: string;
}

/** Matches the server's per-message limit. */
export const ASSISTANT_MAX_CHARS = 1000;

const ENDPOINT = '/api/assistant';
const AVAILABILITY_TIMEOUT_MS = 8000;
/** Covers the provider timeout plus one retry on the server. */
const REPLY_TIMEOUT_MS = 60000;

export type AssistantErrorKind = 'unavailable' | 'rate_limited' | 'invalid' | 'timeout' | 'failed';

export class AssistantError extends Error {
  readonly kind: AssistantErrorKind;

  constructor(kind: AssistantErrorKind, message: string) {
    super(message);
    this.name = 'AssistantError';
    this.kind = kind;
  }
}

async function request(init: RequestInit, timeoutMs: number, signal?: AbortSignal) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const cancel = () => controller.abort();
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    return await fetch(ENDPOINT, { ...init, signal: controller.signal, cache: 'no-store' });
  } catch (error) {
    if (signal?.aborted) throw error;
    if (timedOut) throw new AssistantError('timeout', 'The assistant took too long to answer.');
    throw new AssistantError('failed', 'Could not reach the assistant.');
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

/** False when the endpoint is missing, unconfigured or unreachable. */
export async function getAssistantAvailability(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await request(
      { headers: { Accept: 'application/json' } },
      AVAILABILITY_TIMEOUT_MS,
      signal,
    );
    if (!response.ok) return false;
    const body: unknown = await response.json();
    return typeof body === 'object' && body !== null && (body as { available?: unknown }).available === true;
  } catch (error) {
    if (signal?.aborted) throw error;
    return false;
  }
}

export async function askAssistant(messages: AssistantTurn[], signal?: AbortSignal): Promise<string> {
  const response = await request(
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ messages }),
    },
    REPLY_TIMEOUT_MS,
    signal,
  );

  let body: { reply?: unknown; message?: unknown } = {};
  try {
    body = await response.json();
  } catch {
    // Non-JSON (e.g. a proxy error page): handled by the status below.
  }

  if (response.ok && typeof body.reply === 'string') return body.reply;

  const message = typeof body.message === 'string' ? body.message : 'The assistant could not answer right now.';
  if (response.status === 503 || response.status === 404) {
    throw new AssistantError('unavailable', 'The assistant is not available on this deployment.');
  }
  if (response.status === 429) throw new AssistantError('rate_limited', message);
  if (response.status >= 400 && response.status < 500) throw new AssistantError('invalid', message);
  throw new AssistantError('failed', message);
}
