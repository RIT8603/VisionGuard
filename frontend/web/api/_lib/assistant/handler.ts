/**
 * HTTP handling for /api/assistant, written against the Web Request/Response
 * API so the same code runs as a Vercel Function and in the Vite dev server.
 *
 * GET  -> { available: boolean }
 * POST { messages: ChatTurn[] } -> { reply } | { error, message }
 *
 * The portal is publicly reachable, so every limit here is deliberately tight.
 */

import { SYSTEM_PROMPT } from './knowledge.js';
import { getProvider, ProviderError, type ChatTurn, type Env } from './provider.js';

// ---- request limits ----
const MAX_BODY_BYTES = 16_000;
const MAX_MESSAGE_CHARS = 1_000;
/** Only the most recent turns are sent upstream; older ones are dropped. */
const MAX_TURNS = 8;
const MAX_OUTPUT_TOKENS = 1_500;

// ---- rate limits (per server instance; see README for a platform-level limit) ----
const PER_CLIENT_PER_MINUTE = 5;
const PER_CLIENT_PER_HOUR = 30;
const PER_INSTANCE_PER_HOUR = 150;

const REFUSED_REPLY =
  "Sorry, I can't help with that. I can answer questions about VisionGuard's modules, camera setup and troubleshooting.";

type ErrorCode = 'invalid_request' | 'forbidden' | 'rate_limited' | 'unavailable' | 'upstream';

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers },
  });
}

function fail(status: number, error: ErrorCode, message: string, headers?: Record<string, string>) {
  return json({ error, message }, status, headers);
}

// ---------- rate limiting ----------

const ONE_MINUTE = 60_000;
const ONE_HOUR = 60 * ONE_MINUTE;
const clientHits = new Map<string, number[]>();
let instanceHits: number[] = [];

/** Seconds until the caller may retry, or 0 when the request is allowed. */
function rateLimit(clientId: string, now: number): number {
  instanceHits = instanceHits.filter((t) => now - t < ONE_HOUR);
  const hits = (clientHits.get(clientId) ?? []).filter((t) => now - t < ONE_HOUR);
  const lastMinute = hits.filter((t) => now - t < ONE_MINUTE);

  const wait = (oldest: number, window: number) => Math.ceil((oldest + window - now) / 1000);
  if (instanceHits.length >= PER_INSTANCE_PER_HOUR) return wait(instanceHits[0], ONE_HOUR);
  if (hits.length >= PER_CLIENT_PER_HOUR) return wait(hits[0], ONE_HOUR);
  if (lastMinute.length >= PER_CLIENT_PER_MINUTE) return wait(lastMinute[0], ONE_MINUTE);

  hits.push(now);
  instanceHits.push(now);
  clientHits.set(clientId, hits);
  if (clientHits.size > 5_000) {
    for (const [id, times] of clientHits) {
      if (!times.some((t) => now - t < ONE_HOUR)) clientHits.delete(id);
    }
  }
  return 0;
}

function clientIdOf(request: Request): string {
  const forwarded = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim();
  return forwarded || request.headers.get('x-real-ip') || 'unknown';
}

// ---------- validation ----------

function parseMessages(body: unknown): ChatTurn[] | string {
  if (typeof body !== 'object' || body === null || !Array.isArray((body as { messages?: unknown }).messages)) {
    return 'Expected { messages: [...] }.';
  }
  const raw = (body as { messages: unknown[] }).messages;
  if (raw.length === 0) return 'messages is empty.';

  const turns: ChatTurn[] = [];
  for (const item of raw) {
    if (typeof item !== 'object' || item === null) return 'Each message must be an object.';
    const { role, content } = item as { role?: unknown; content?: unknown };
    if (role !== 'user' && role !== 'assistant') return 'role must be "user" or "assistant".';
    if (typeof content !== 'string' || !content.trim()) return 'content must be a non-empty string.';
    if (content.length > MAX_MESSAGE_CHARS) {
      return `Messages are limited to ${MAX_MESSAGE_CHARS} characters.`;
    }
    turns.push({ role, content: content.trim() });
  }

  const recent = turns.slice(-MAX_TURNS);
  while (recent.length && recent[0].role !== 'user') recent.shift();
  if (!recent.length || recent[recent.length - 1].role !== 'user') {
    return 'The last message must be from the user.';
  }
  return recent;
}

/** Browsers send Origin on cross-site POSTs; only this site may call the endpoint. */
function sameOrigin(request: Request): boolean {
  const origin = request.headers.get('origin');
  if (!origin) return true;
  try {
    return new URL(origin).host === (request.headers.get('host') ?? new URL(request.url).host);
  } catch {
    return false;
  }
}

/**
 * The panel shows replies as plain text, and models add markdown anyway.
 * Strips the common marks and keeps line breaks and list structure.
 * Underscores are left alone: module names such as person_detection use them.
 */
export function toPlainText(text: string): string {
  return text
    .replace(/^[ \t]*```[^\n]*\n?/gm, '') // code fence lines (the code inside stays)
    .replace(/^[ \t]*#{1,6}[ \t]+/gm, '') // heading markers
    .replace(/^([ \t]*)[*+][ \t]+/gm, '$1- ') // "* item" bullets -> "- item"
    .replace(/(?<!\*)\*\*(?=[^\s*])([^*\n]*?[^\s*])\*\*(?!\*)/g, '$1') // **bold**, not "***"
    .replace(/(^|[\s(])\*([^\s*][^*\n]*?)\*(?=[\s.,;:!?)]|$)/gm, '$1$2') // *italic*
    .replace(/`([^`\n]+)`/g, '$1') // `inline code`
    .replace(/^([ \t]*\d+)\\+\./gm, '$1.') // "1\." numbered-list escape
    // Escaped punctuation ("\-", "\_", "\*", or doubled "\\-") -> the character
    // itself. Last, so escaped marks stay literal. "\." after a letter or digit
    // is kept: that is a Windows path such as frontend\web\.env.local.
    .replace(/\\+([!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~])/g, (match, char: string, offset: number, all: string) =>
      char === '.' && /[A-Za-z0-9]/.test(all[offset - 1] ?? '') ? match : char,
    )
    .trim();
}

// ---------- handlers ----------

export async function handleGet(env: Env): Promise<Response> {
  return json({ available: (await getProvider(env)) !== null });
}

export async function handlePost(request: Request, env: Env): Promise<Response> {
  if (!sameOrigin(request)) return fail(403, 'forbidden', 'Cross-origin requests are not allowed.');
  if (!request.headers.get('content-type')?.includes('application/json')) {
    return fail(415, 'invalid_request', 'Send JSON.');
  }

  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) return fail(413, 'invalid_request', 'Request is too large.');

  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return fail(400, 'invalid_request', 'Request is not valid JSON.');
  }
  const messages = parseMessages(body);
  if (typeof messages === 'string') return fail(400, 'invalid_request', messages);

  const provider = await getProvider(env);
  if (!provider) return fail(503, 'unavailable', 'The assistant is not configured.');

  const retryAfter = rateLimit(clientIdOf(request), Date.now());
  if (retryAfter > 0) {
    return fail(429, 'rate_limited', 'Too many questions. Please wait a moment.', {
      'Retry-After': String(retryAfter),
    });
  }

  try {
    const reply = await provider.reply({
      system: SYSTEM_PROMPT,
      messages,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
    });
    return json({ reply: toPlainText(reply) });
  } catch (error) {
    if (error instanceof ProviderError) {
      if (error.kind === 'refused') return json({ reply: REFUSED_REPLY });
      // Logged without the user's text; the client gets no provider detail.
      console.error(`[assistant] ${provider.name}: ${error.kind}: ${error.message}`);
      if (error.kind === 'rate_limited') {
        return fail(429, 'rate_limited', 'The assistant is busy. Please try again shortly.', {
          'Retry-After': '30',
        });
      }
      return fail(502, 'upstream', 'The assistant could not answer right now.');
    }
    console.error(`[assistant] ${provider.name}: unexpected error`, error instanceof Error ? error.name : '');
    return fail(502, 'upstream', 'The assistant could not answer right now.');
  }
}
