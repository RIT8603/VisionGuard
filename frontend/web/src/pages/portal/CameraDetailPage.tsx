import { Link, useParams } from 'react-router';
import { PATHS } from '../../app/paths';
import { EventFeed } from '../../components/pipeline/EventFeed';
import { FrameResults } from '../../components/pipeline/FrameResults';
import { ModulesPanel } from '../../components/pipeline/ModulesPanel';
import { StatsPanel } from '../../components/pipeline/StatsPanel';
import { Panel } from '../../components/ui/Panel';
import { findCamera, useCameraRegistry } from '../../hooks/useCameraRegistry';
import { DASH, fmtTime } from '../../lib/format';

const PENDING = 'Camera-specific backend data is not available yet.';

/**
 * One camera: identity, stream, telemetry, modules, frame results and events.
 *
 * Mirrors the Live Monitoring layout with the same panels. They are fed
 * nothing for now: camera-scoped status, events and stream endpoints do not
 * exist, and the single pipeline's data must not be shown as this camera's.
 * No stream connection is opened here.
 */
export function CameraDetailPage() {
  const { cameraId = '' } = useParams();
  const registry = useCameraRegistry();
  const camera = findCamera(registry, cameraId);
  const notFound = registry.state === 'ready' && camera === null;

  const identity: [label: string, value: string][] = [
    ['Requested id', cameraId],
    ['Name', camera?.name ?? DASH],
    ['Source', camera?.source_type ?? DASH],
    ['Status', camera?.status ?? DASH],
    ['Last frame', camera?.last_frame_at ? fmtTime(camera.last_frame_at) : DASH],
  ];

  return (
    <>
      <title>{`Camera ${cameraId} · VisionGuard`}</title>

      <div className="page page-tight">
        <Link className="panel-link" to={PATHS.portal.cameras}>
          ← All cameras
        </Link>
        <header className="page-head">
          <h2 className="page-title">{camera?.name ?? 'Camera'}</h2>
          <p className="page-intro mono camera-id-line">{cameraId}</p>
        </header>

        {notFound ? (
          <div className="notice notice-warn" role="status">
            <span className="notice-word">Unknown</span>
            <div className="notice-body">
              The backend has no camera with id <code>{cameraId}</code>.
            </div>
          </div>
        ) : (
          <div className="notice notice-info" role="note">
            <span className="notice-word">Pending</span>
            <div className="notice-body">
              <strong>{PENDING}</strong> The backend has no camera registry or camera-scoped routes,
              so this camera id cannot be resolved. The backend’s single pipeline is on{' '}
              <Link to={PATHS.portal.live}>Live Monitoring</Link>.
            </div>
          </div>
        )}
      </div>

      <div className="layout">
        <div className="column column-main">
          <section className="feed" aria-label="Camera stream">
            <header className="feed-bar">
              <span className="feed-title">Live feed</span>
              <span className="mono feed-source">{cameraId}</span>
            </header>
            <div className="stream" style={{ aspectRatio: '16 / 9' }}>
              <div className="stream-overlay">No camera stream available</div>
            </div>
          </section>
          <StatsPanel stats={null} eventCount={null} />
          <FrameResults frame={null} emptyMessage={PENDING} />
        </div>
        <div className="column column-side">
          <Panel title="Camera">
            <dl className="rows">
              {identity.map(([label, value]) => (
                <div className="row" key={label}>
                  <dt>{label}</dt>
                  <dd className="mono">{value}</dd>
                </div>
              ))}
            </dl>
          </Panel>
          <ModulesPanel
            modules={[]}
            latency={undefined}
            errors={undefined}
            live={false}
            emptyMessage={PENDING}
          />
          <EventFeed events={[]} error={null} emptyMessage="Camera-scoped events need backend support." />
        </div>
      </div>
    </>
  );
}
