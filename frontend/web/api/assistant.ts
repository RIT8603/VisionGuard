/**
 * Vercel Function: POST /api/assistant (chat), GET /api/assistant (availability).
 * Logic lives in _lib/assistant/ so the Vite dev server can reuse it.
 */

import { handleGet, handlePost } from './_lib/assistant/handler.js';

export function GET(): Promise<Response> {
  return handleGet(process.env);
}

export function POST(request: Request): Promise<Response> {
  return handlePost(request, process.env);
}
