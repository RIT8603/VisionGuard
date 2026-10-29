import { useCallback, useEffect, useRef } from 'react';
import { streamUrl } from '../../api/endpoints';
import type { CameraStatus } from '../../api/types';
import { useMjpegStream } from '../../hooks/useMjpegStream';
import { DASH, fmtDuration } from '../../lib/format';
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
 * The backend's MJPEG stream, read with fetch and drawn on a canvas (see
 * useMjpegStream). Boxes, track ids and zones are already drawn into the
 * frames server-side.
 *
 * One connection per (camera, backend run, visibility): re-rendering never
 * reopens it. A connection that goes quiet is replaced within seconds, and
 * the last frame is dimmed and labelled while that happens, so a stale
 * picture never passes for live video.
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
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const connectionKey = `${restartCount}:${camera ?? ''}`;
  const nonce = useRef(0);
  const makeUrl = useCallback(() => streamUrl(++nonce.current, camera, CLIENT_ID), [camera]);
  const stream = useMjpegStream(makeUrl, shouldStream, canvasRef, connectionKey);

  // A different camera must not show the previous camera's last frame.
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  }, [camera]);

  const stalled = shouldStream && stream.stalledMs > 0;
  const cameraOverlay =
    cameraStatus === 'failed'
      ? `This camera failed: ${cameraError ?? 'see the Cameras page'}`
      : cameraStatus === 'offline'
        ? 'This camera has stopped (its video ended or it was switched off).'
        : cameraStatus === 'starting' && !stream.hasFrame
          ? 'Camera starting…'
          : null;

  const overlay =
    cameraOverlay ??
    (shouldStream && !stream.hasFrame
      ? 'Connecting to camera…'
      : stalled && stream.stalledMs > 5000
        ? `No new video for ${fmtDuration(stream.stalledMs / 1000)} — reconnecting…`
        : OVERLAY_TEXT[viewState]);

  const aspectRatio = resolution ? `${resolution.width} / ${resolution.height}` : '16 / 9';

  return (
    <section className="feed" aria-label="Live camera feed">
      <header className="feed-bar">
        <span className="feed-title">Live feed</span>
        <span className="mono feed-source">{sourceLabel ?? DASH}</span>
        <span className="spacer" />
        {shouldStream && stream.hasFrame && (
          <span className="mono muted" title="Frames drawn in this browser per second">
            {stream.shownFps.toFixed(0)} fps shown
          </span>
        )}
        {resolution && (
          <span className="mono muted">
            {resolution.width}×{resolution.height}
          </span>
        )}
        <StateBadge state={viewState} />
      </header>
      <div className={`stream${stalled ? ' is-stalled' : ''}`} style={{ aspectRatio }}>
        <canvas
          ref={canvasRef}
          className="stream-canvas"
          role="img"
          aria-label="Live camera view with detections drawn by the backend"
        />
        {overlay && (
          <div className={`stream-overlay${cameraDown ? ' is-failed' : ''}`}>{overlay}</div>
        )}
        {stalled && !overlay && (
          <span className="stream-tag">Delayed {fmtDuration(stream.stalledMs / 1000)} · reconnecting</span>
        )}
        {viewState === 'stale' && !stalled && <span className="stream-tag">Stale</span>}
      </div>
    </section>
  );
}
