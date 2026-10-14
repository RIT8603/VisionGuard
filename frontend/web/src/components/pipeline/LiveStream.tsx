import { useEffect, useRef, useState } from 'react';
import { streamUrl } from '../../api/endpoints';
import type { CameraStatus } from '../../api/types';
import { STREAM_HOLD_MS, STREAM_RETRY_MS } from '../../config';
import { DASH } from '../../lib/format';
import { type ViewState } from '../../state/viewState';
import { StateBadge } from './StateBadge';

interface LiveStreamProps {
  viewState: ViewState;
  /** False while the tab is hidden: the connection is closed, not left streaming. */
  active: boolean;
  /** A change means the backend restarted and the old connection is dead. */
  restartCount: number;
  /** Frame size, to reserve the right aspect ratio before the first image. */
  resolution?: { width: number; height: number };
  /** Shown in the feed's title bar, e.g. the source id. */
  sourceLabel?: string;
  /** Which camera to stream (null = the backend's first). */
  camera?: string | null;
  /** The selected camera's own state, when several cameras run. */
  cameraStatus?: CameraStatus;
  /** Why the selected camera failed, if it did. */
  cameraError?: string | null;
}

const BLANK_IMAGE = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

const CLIENT_ID =
  typeof window !== 'undefined'
    ? ((window as any).__VG_CLIENT_ID ??= Math.random().toString(36).slice(2, 10))
    : 'web-client';

const OVERLAY_TEXT: Partial<Record<ViewState, string>> = {
  loading: 'Connecting to backend…',
  offline: 'Backend offline',
  stopped: 'Pipeline stopped',
  waiting: 'Waiting for the first frame…',
};

/**
 * The backend's MJPEG stream in a plain <img>. Boxes, track ids and zones are
 * already drawn into the frames server-side.
 *
 * The <img> sits on top of a <canvas> that holds a copy of the last frame it
 * showed. Reconnecting replaces the <img>, and a new one is empty until its
 * first frame arrives; without the canvas the feed went black for that long
 * every time. With it the picture simply holds until the next frame.
 *
 * Only one of these should be mounted per tab: each holds an HTTP connection
 * open for as long as it is rendered.
 */
export function LiveStream({
  viewState,
  active,
  restartCount,
  resolution,
  sourceLabel,
  camera = null,
  cameraStatus,
  cameraError = null,
}: LiveStreamProps) {
  // A failed or stopped camera has no stream to open: the backend ends the
  // connection at once. Say what is actually wrong with that camera instead.
  const cameraDown = cameraStatus === 'failed' || cameraStatus === 'offline';
  const shouldStream = active && viewState !== 'offline' && viewState !== 'stopped' && !cameraDown;
  const [nonce, setNonce] = useState(0);
  const [broken, setBroken] = useState(false);
  const [held, setHeld] = useState(false);
  const [isSwitching, setIsSwitching] = useState(false);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Open a fresh connection whenever streaming (re)starts or the backend was replaced.
  const connectionKey = `${shouldStream}:${restartCount}:${camera ?? ''}`;
  const [lastKey, setLastKey] = useState(connectionKey);
  if (connectionKey !== lastKey) {
    setLastKey(connectionKey);
    if (shouldStream) {
      setNonce((n) => n + 1);
      setBroken(false);
    }
  }

  // After an <img> error, retry while the backend still answers status polls.
  useEffect(() => {
    if (!broken || !shouldStream) return;
    const timer = window.setTimeout(() => {
      setBroken(false);
      setNonce((n) => n + 1);
    }, STREAM_RETRY_MS);
    return () => window.clearTimeout(timer);
  }, [broken, shouldStream]);

  // Automatically clear broken / reconnecting state as soon as status indicates live
  useEffect(() => {
    if (viewState === 'live') {
      if (broken) setBroken(false);
      if (isSwitching) setIsSwitching(false);
    }
  }, [viewState, broken, isSwitching]);

  // Keep the canvas a recent copy of what the <img> shows.
  useEffect(() => {
    if (!shouldStream) return;
    const timer = window.setInterval(() => {
      const img = imgRef.current;
      const canvas = canvasRef.current;
      if (!img || !canvas || !img.naturalWidth) return;
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      try {
        canvas.getContext('2d')?.drawImage(img, 0, 0);
        setHeld(true);
      } catch {
        // An image that failed mid-decode cannot be drawn; keep the last copy.
      }
    }, STREAM_HOLD_MS);
    return () => window.clearInterval(timer);
  }, [shouldStream]);

  // Cleanly terminate old stream connection on camera switch, clear held canvas and mark switching
  useEffect(() => {
    setIsSwitching(true);
    setBroken(false);
    setHeld(false);
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    setNonce((n) => n + 1);

    return () => {
      if (imgRef.current) {
        imgRef.current.onload = null;
        imgRef.current.onerror = null;
        imgRef.current.src = BLANK_IMAGE;
      }
    };
  }, [camera]);

  const reconnecting = broken && shouldStream;
  const cameraOverlay =
    cameraStatus === 'failed'
      ? `This camera failed: ${cameraError ?? 'see the Cameras page'}`
      : cameraStatus === 'offline'
        ? 'This camera has stopped (its video ended or it was switched off).'
        : cameraStatus === 'starting' && !held
          ? 'Camera starting…'
          : null;

  const overlay =
    cameraOverlay ??
    (isSwitching && !held
      ? 'Connecting to camera…'
      : reconnecting && !held
        ? 'Stream interrupted — reconnecting…'
        : OVERLAY_TEXT[viewState]);

  const aspectRatio = resolution ? `${resolution.width} / ${resolution.height}` : '16 / 9';

  return (
    <section className="feed" aria-label="Live camera feed">
      <header className="feed-bar">
        <span className="feed-title">Live feed</span>
        <span className="mono feed-source">{sourceLabel ?? DASH}</span>
        <span className="spacer" />
        {resolution && (
          <span className="mono muted">
            {resolution.width}×{resolution.height}
          </span>
        )}
        <StateBadge state={viewState} />
      </header>
      <div className="stream" style={{ aspectRatio }}>
        <canvas ref={canvasRef} className="stream-held" aria-hidden="true" />
        {shouldStream && (
          <img
            key={`${camera ?? 'default'}:${nonce}`}
            ref={imgRef}
            src={streamUrl(nonce, camera, CLIENT_ID)}
            alt="Live camera view with detections drawn by the backend"
            onLoad={() => {
              setIsSwitching(false);
              setBroken(false);
            }}
            onError={() => {
              if (!isSwitching) {
                setBroken(true);
              }
            }}
          />
        )}
        {overlay && (
          <div className={`stream-overlay${cameraDown ? ' is-failed' : ''}`}>{overlay}</div>
        )}
        {reconnecting && held && <span className="stream-tag">Reconnecting</span>}
        {viewState === 'stale' && !reconnecting && <span className="stream-tag">Stale</span>}
      </div>
    </section>
  );
}
