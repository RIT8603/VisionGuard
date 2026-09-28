import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 127.0.0.1 rather than "localhost": Node can resolve localhost to ::1, while
// the backend binds IPv4 127.0.0.1 only by default.
const backend = process.env.VG_BACKEND ?? 'http://127.0.0.1:8080';

// Only the API and the stream are proxied. "/" stays with Vite; on the backend
// it serves the legacy frontend/static page.
const proxy = {
  '/api': { target: backend, changeOrigin: true },
  '/stream.mjpg': { target: backend, changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
});
