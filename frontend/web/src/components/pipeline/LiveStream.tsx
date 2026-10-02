import { useEffect, useState } from 'react';
import { streamUrl } from '../../api/endpoints';
import { STREAM_RETRY_MS } from '../../config';
import { DASH } from '../../lib/format';
import { isReachable, type ViewState } from '../../state/viewState';
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
}

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
 * Only one of these should be mounted per tab: each holds an HTTP connection
 * open for as long as it is rendered.
 */
export function LiveStream({
  viewState,
  active,
  restartCount,
  resolution,
  sourceLabel,
}: LiveStreamProps) {
  const shouldStream = active && isReachable(viewState);
  const [nonce, setNonce] = useState(0);
  const [broken, setBroken] = useState(false);

  // Open a fresh connection whenever streaming (re)starts or the backend was
  // replaced. Adjusting state during render (rather than in an effect) means
  // the <img> mounts once, with the new URL, instead of connecting twice.
  const connectionKey = `${shouldStream}:${restartCount}`;
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

  const overlay = broken && shouldStream ? 'Stream interrupted — reconnecting…' : OVERLAY_TEXT[viewState];
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
        {shouldStream && !broken && (
          <img
            key={nonce}
            src={streamUrl(nonce)}
            alt="Live camera view with detections drawn by the backend"
            onError={() => setBroken(true)}
          />
        )}
        {overlay && <div className="stream-overlay">{overlay}</div>}
        {viewState === 'stale' && <span className="stream-tag">Stale</span>}
      </div>
    </section>
  );
}
