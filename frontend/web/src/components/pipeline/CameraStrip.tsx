import { useEffect, useState } from 'react';
import { snapshotUrl } from '../../api/endpoints';
import type { CameraInfo } from '../../api/types';
import { TILE_REFRESH_MS } from '../../config';
import { DASH } from '../../lib/format';
import { CameraStatusBadge } from './CameraStatusBadge';

interface CameraStripProps {
  cameras: CameraInfo[];
  /** The camera the main feed is following. */
  selected: string | null;
  onSelect: (id: string) => void;
  /** False while the tab is hidden: thumbnails stop refreshing. */
  active: boolean;
}

/** Counts up every TILE_REFRESH_MS while `active`; the thumbnails' cache-buster. */
function useTick(active: boolean): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setTick((n) => n + 1), TILE_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [active]);
  return tick;
}

/**
 * One thumbnail per camera; click one to follow it in the main feed.
 *
 * Thumbnails are short snapshot requests, not MJPEG streams: a browser allows
 * only about six connections per host, so one stream per camera would starve
 * the status and event polling as soon as there were more than a few.
 */
export function CameraStrip({ cameras, selected, onSelect, active }: CameraStripProps) {
  const tick = useTick(active);

  return (
    <section className="camera-strip" aria-label="Cameras">
      <ul className="camera-tiles">
        {cameras.map((camera) => {
          const isSelected = camera.id === selected;
          const hasImage = camera.status === 'live' || camera.status === 'stalled';
          return (
            <li key={camera.id}>
              <button
                type="button"
                className={isSelected ? 'camera-tile is-selected' : 'camera-tile'}
                onClick={() => onSelect(camera.id)}
                aria-pressed={isSelected}
                title={camera.error ?? undefined}
              >
                <span className="camera-tile-image">
                  {hasImage ? (
                    <img
                      src={snapshotUrl(camera.id, tick)}
                      alt={`Latest frame from ${camera.name}`}
                      loading="eager"
                    />
                  ) : (
                    <span className="camera-tile-empty">
                      {camera.status === 'failed' ? 'Failed' : camera.status === 'starting' ? 'Starting…' : 'No video'}
                    </span>
                  )}
                </span>
                <span className="camera-tile-bar">
                  <span className="camera-tile-name">{camera.name}</span>
                  <CameraStatusBadge status={camera.status} />
                </span>
                <span className="camera-tile-meta mono">
                  {camera.status === 'failed' && camera.error
                    ? camera.error
                    : `${camera.fps === null ? DASH : camera.fps.toFixed(1)} fps · ${camera.event_count} events`}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
