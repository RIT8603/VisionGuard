import type { IncomingMessage } from 'node:http';
import { defineConfig, loadEnv, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// 127.0.0.1 rather than "localhost": Node can resolve localhost to ::1,
// while the backend binds IPv4 127.0.0.1 only by default.
const backend = process.env.VG_BACKEND ?? 'http://127.0.0.1:8080';

// Only the API and the stream are proxied.
const proxy = {
  '/api': {
    target: backend,
    changeOrigin: true,
  },
  '/stream.mjpg': {
    target: backend,
    changeOrigin: true,
  },
};

const MAX_DEV_BODY_BYTES = 64_000;

function toWebRequest(req: IncomingMessage, body: Buffer): Request {
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (typeof value === 'string') headers.set(key, value);
    else if (Array.isArray(value)) headers.set(key, value.join(', '));
  }
  if (!headers.has('x-forwarded-for') && req.socket.remoteAddress) {
    headers.set('x-forwarded-for', req.socket.remoteAddress);
  }
  return new Request(`http://${req.headers.host ?? 'localhost'}/api/assistant`, {
    method: 'POST',
    headers,
    body: new Uint8Array(body),
  });
}

/**
 * Dev only: serves /api/assistant (a Vercel Function in production) from the
 * same handler, ahead of the /api proxy to the Python backend. Server-side
 * env (OPENROUTER_API_KEY etc.) comes from .env.local; Vite exposes only
 * VITE_-prefixed names to the browser, so the key never reaches it.
 */
function assistantDevApi(): Plugin {
  let env: Record<string, string> = {};
  return {
    name: 'visionguard-assistant-dev-api',
    apply: 'serve',
    configResolved(config) {
      env = loadEnv(config.mode, config.envDir, '');
    },
    configureServer(server) {
      server.middlewares.use('/api/assistant', async (req, res, next) => {
        if (req.url !== '/' && !req.url?.startsWith('/?')) return next();
        try {
          const handler = (await server.ssrLoadModule(
            '/api/_lib/assistant/handler.ts',
          )) as typeof import('./api/_lib/assistant/handler.ts');

          let response: Response;
          if (req.method === 'GET') {
            response = await handler.handleGet(env);
          } else if (req.method === 'POST') {
            const chunks: Buffer[] = [];
            let size = 0;
            for await (const chunk of req) {
              size += (chunk as Buffer).length;
              if (size > MAX_DEV_BODY_BYTES) break;
              chunks.push(chunk as Buffer);
            }
            response =
              size > MAX_DEV_BODY_BYTES
                ? new Response(null, { status: 413 })
                : await handler.handlePost(toWebRequest(req, Buffer.concat(chunks)), env);
          } else {
            response = new Response(null, { status: 405, headers: { Allow: 'GET, POST' } });
          }

          res.statusCode = response.status;
          response.headers.forEach((value, key) => res.setHeader(key, value));
          res.end(Buffer.from(await response.arrayBuffer()));
        } catch (error) {
          next(error);
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), assistantDevApi()],

  server: {
    host: '0.0.0.0',
    // A leading dot allows the domain and every subdomain: a Cloudflare quick
    // tunnel gets a new random name each time it starts.
    allowedHosts: [
      'desktop-ikehftb.taile5cc80.ts.net',
      'paracosm13.tail3291a7.ts.net',
      '.trycloudflare.com',
    ],
    proxy,
  },

  preview: {
    proxy,
  },
});