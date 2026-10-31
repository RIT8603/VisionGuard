import { useEffect, useRef, useState, type RefObject } from 'react';
import { STREAM_LOG_MS, STREAM_MAX_BACKOFF_MS, STREAM_STALL_MS } from '../config';

/** What the viewer is actually getting, measured in the browser. */
export interface StreamStats {
  /** A response is open (frames may still be slow). */
  connected: boolean;
  /** At least one frame has been drawn since the component mounted. */
  hasFrame: boolean;
  /** Milliseconds since the last new frame arrived (0 while frames flow). */
  stalledMs: number;
  /** Frames received / drawn per second over the last log window. */
  receivedFps: number;
  shownFps: number;
  /** Backend sequence number of the frame on screen. */
  seq: number | null;
  reconnects: number;
}

const INITIAL: StreamStats = {
  connected: false,
  hasFrame: false,
  stalledMs: 0,
  receivedFps: 0,
  shownFps: 0,
  seq: null,
  reconnects: 0,
};

const CRLFCRLF = [13, 10, 13, 10];

function indexOfSeq(buf: Uint8Array, seq: number[], from = 0): number {
  outer: for (let i = from; i <= buf.length - seq.length; i++) {
    for (let j = 0; j < seq.length; j++) if (buf[i + j] !== seq[j]) continue outer;
    return i;
  }
  return -1;
}

const header = (text: string, name: string): number | null => {
  const m = new RegExp(`^${name}:\\s*([0-9.]+)`, 'mi').exec(text);
  return m ? Number(m[1]) : null;
};

/**
 * Reads the backend's MJPEG stream with fetch and draws each frame on a canvas.
 *
 * Unlike an <img>, this knows when every frame arrives: it reconnects when no
 * frame has come for STREAM_STALL_MS, reports a stalled picture instead of
 * leaving it looking live, draws only the newest frame when decoding falls
 * behind, and logs per-frame sequence numbers so delivery can be traced from
 * the backend's X-Frame-* headers to the screen.
 *
 * `makeUrl` is called once per connection attempt, never per render.
 */
export function useMjpegStream(
  makeUrl: () => string,
  enabled: boolean,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  connectionKey: string,
): StreamStats {
  const [stats, setStats] = useState<StreamStats>(INITIAL);
  const makeUrlRef = useRef(makeUrl);
  makeUrlRef.current = makeUrl;

  useEffect(() => {
    if (!enabled) {
      setStats((s) => ({ ...s, connected: false, stalledMs: 0 }));
      return;
    }
    let stopped = false;
    let controller: AbortController | null = null;
    let lastFrameAt = performance.now();
    let failures = 0;
    let reconnects = 0;
    let hasFrame = false;
    let connected = false;
    let shownSeq: number | null = null;
    // Newest frame waiting to be drawn; older ones are dropped, never queued.
    let pending: { bytes: Uint8Array; seq: number | null } | null = null;
    let drawing = false;
    const win = { received: 0, shown: 0, dropped: 0, maxGap: 0, firstSeq: null as number | null, lastSeq: null as number | null, ageMs: 0 };
    let windowStart = performance.now();
    let lastArrival = 0;

    const draw = async () => {
      if (drawing) return;
      drawing = true;
      while (pending && !stopped) {
        const frame = pending;
        pending = null;
        try {
          const bitmap = await createImageBitmap(new Blob([frame.bytes as BlobPart], { type: 'image/jpeg' }));
          const canvas = canvasRef.current;
          if (canvas && !stopped) {
            if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
              canvas.width = bitmap.width;
              canvas.height = bitmap.height;
            }
            canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
            win.shown += 1;
            shownSeq = frame.seq;
            hasFrame = true;
          }
          bitmap.close();
        } catch {
          // A corrupt frame: skip it, the next one replaces it anyway.
        }
      }
      drawing = false;
    };

    const onFrame = (bytes: Uint8Array, head: string) => {
      const now = performance.now();
      const seq = header(head, 'X-Frame-Seq');
      const sent = header(head, 'X-Frame-Sent');
      const captured = header(head, 'X-Frame-Captured');
      if (lastArrival) win.maxGap = Math.max(win.maxGap, now - lastArrival);
      lastArrival = now;
      lastFrameAt = now;
      failures = 0;
      win.received += 1;
      if (win.firstSeq === null) win.firstSeq = seq;
      win.lastSeq = seq;
      if (sent !== null && captured !== null) win.ageMs = Math.max(win.ageMs, (sent - captured) * 1000);
      if (pending) win.dropped += 1;
      pending = { bytes, seq };
      void draw();
    };

    const read = async (signal: AbortSignal) => {
      const response = await fetch(makeUrlRef.current(), { signal, cache: 'no-store', credentials: 'same-origin' });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      connected = true;
      const reader = response.body.getReader();
      let buf = new Uint8Array(0);
      for (;;) {
        const { value, done } = await reader.read();
        if (done) throw new Error('stream ended');
        const next = new Uint8Array(buf.length + value.length);
        next.set(buf);
        next.set(value, buf.length);
        buf = next;
        for (;;) {
          const end = indexOfSeq(buf, CRLFCRLF);
          if (end < 0) break;
          const head = new TextDecoder('latin1').decode(buf.subarray(0, end));
          const length = header(head, 'Content-Length');
          if (length === null) {
            buf = buf.subarray(end + 4); // not a part header: resynchronise
            continue;
          }
          const start = end + 4;
          if (buf.length < start + length) break;
          onFrame(buf.slice(start, start + length), head);
          buf = buf.subarray(start + length);
        }
      }
    };

    const connect = async () => {
      while (!stopped) {
        controller = new AbortController();
        lastFrameAt = performance.now();
        try {
          await read(controller.signal);
        } catch (error) {
          if (stopped) return;
          console.info('[vg-stream] connection ended:', error instanceof Error ? error.message : String(error));
        }
        connected = false;
        if (stopped) return;
        reconnects += 1;
        failures += 1;
        // First retry at once (a stall is usually one bad connection), then back off.
        const delay = failures <= 1 ? 0 : Math.min(STREAM_MAX_BACKOFF_MS, 500 * 2 ** (failures - 2));
        if (delay) await new Promise((resolve) => window.setTimeout(resolve, delay));
      }
    };

    // Watchdog: a silent connection is replaced, not trusted.
    const watchdog = window.setInterval(() => {
      const stalledMs = performance.now() - lastFrameAt;
      if (connected && stalledMs > STREAM_STALL_MS) {
        console.info(`[vg-stream] no frame for ${Math.round(stalledMs)} ms; reconnecting`);
        controller?.abort();
      }
      setStats((s) => {
        const next = {
          ...s,
          connected,
          hasFrame,
          stalledMs: hasFrame && stalledMs > 1500 ? stalledMs : 0,
          seq: shownSeq,
          reconnects,
        };
        return next.connected === s.connected && next.hasFrame === s.hasFrame && next.stalledMs === s.stalledMs &&
          next.seq === s.seq && next.reconnects === s.reconnects ? s : next;
      });
    }, 500);

    // Per-window delivery figures, mirroring the backend's "stream ..." log lines.
    const logger = window.setInterval(() => {
      const span = (performance.now() - windowStart) / 1000;
      const receivedFps = win.received / span;
      const shownFps = win.shown / span;
      console.info(
        `[vg-stream] seq ${win.firstSeq ?? '-'}-${win.lastSeq ?? '-'}: received ${receivedFps.toFixed(1)} fps, ` +
          `shown ${shownFps.toFixed(1)} fps, dropped ${win.dropped}, max gap ${Math.round(win.maxGap)} ms, ` +
          `max capture->send ${Math.round(win.ageMs)} ms, reconnects ${reconnects}`,
      );
      (window as unknown as { __vgStream?: unknown }).__vgStream = {
        receivedFps, shownFps, maxGapMs: win.maxGap, dropped: win.dropped, reconnects, seq: shownSeq, at: Date.now(),
      };
      setStats((s) => ({ ...s, receivedFps, shownFps }));
      Object.assign(win, { received: 0, shown: 0, dropped: 0, maxGap: 0, firstSeq: null, lastSeq: null, ageMs: 0 });
      windowStart = performance.now();
    }, STREAM_LOG_MS);

    void connect();
    return () => {
      stopped = true;
      controller?.abort();
      window.clearInterval(watchdog);
      window.clearInterval(logger);
    };
  }, [enabled, connectionKey, canvasRef]);

  return stats;
}
