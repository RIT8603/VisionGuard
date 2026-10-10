import { defineConfig } from 'vite';
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

export default defineConfig({
  plugins: [react()],

  server: {
    host: '0.0.0.0',
    allowedHosts: ['desktop-ikehftb.taile5cc80.ts.net'],
    proxy,
  },

  preview: {
    proxy,
  },
});