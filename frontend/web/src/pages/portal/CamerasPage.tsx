import { CAMERA_ENDPOINTS } from '../../api/cameras';
import { PATHS } from '../../app/paths';
import { CameraCard, CameraCardPreview } from '../../components/cameras/CameraCard';
import { StateBadge } from '../../components/pipeline/StateBadge';
import { Panel } from '../../components/ui/Panel';
import { PanelLink } from '../../components/ui/PanelLink';
import { useCameraRegistry, type CameraRegistry } from '../../hooks/useCameraRegistry';
import { DASH } from '../../lib/format';
import { usePipeline } from '../../state/PipelineProvider';

/** The pending contract, listed so the integration gap is concrete. */
const PENDING_ENDPOINTS = [
  CAMERA_ENDPOINTS.list,
  CAMERA_ENDPOINTS.status('{cameraId}'),
  CAMERA_ENDPOINTS.events('{cameraId}'),
  CAMERA_ENDPOINTS.snapshot('{cameraId}'),
  CAMERA_ENDPOINTS.stream('{cameraId}'),
].map(decodeURIComponent);

/**
 * The multi-camera registry. Lists only what GET /api/cameras returns; while
 * the backend has no such route it shows the integration state instead.
 */
export function CamerasPage() {
  const registry = useCameraRegistry();
  const { status } = usePipeline();
  const info = status.data?.info;

  return (
    <>
      <title>Cameras · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Cameras</h2>
          <p className="page-intro">
            Every camera the backend knows about, with its status, last frame and a snapshot. Open a
            camera for its live stream, telemetry, modules and events.
          </p>
        </header>

        {registry.state === 'unsupported' && (
          <div className="notice notice-info camera-notice" role="note">
            <span className="notice-word">Pending</span>
            <div className="notice-body">
              Camera discovery is not available yet. The backend runs a single pipeline with one
              status feed, one event buffer and one stream, and has no camera registry or
              camera-scoped endpoints. Cameras will appear here once it does.
            </div>
          </div>
        )}

        <div className="cameras-layout">
          <Panel title="Camera registry" aside={registryAside(registry)} className="cameras-registry">
            <RegistryBody registry={registry} />
          </Panel>

          <aside className="cameras-aside" aria-label="Current pipeline">
            {/* Real data from /api/status, deliberately not shown as a registry entry. */}
            <Panel title="Current pipeline source">
              <dl className="rows">
                <div className="row">
                  <dt>State</dt>
                  <dd className="mono">
                    <StateBadge state={status.viewState} />
                  </dd>
                </div>
                <div className="row">
                  <dt>Source id</dt>
                  <dd className="mono">{info?.source_id ?? DASH}</dd>
                </div>
                <div className="row">
                  <dt>Source</dt>
                  <dd className="mono">{info?.source_type ?? DASH}</dd>
                </div>
              </dl>
              <p className="camera-aside-note">
                The one source the backend runs today. It has no camera id, so it is not listed in
                the registry.
              </p>
            </Panel>
            <PanelLink to={PATHS.portal.live}>Live monitoring →</PanelLink>

            {registry.state === 'unsupported' && (
              <Panel title="Pending endpoints">
                <ul className="camera-endpoints mono">
                  {PENDING_ENDPOINTS.map((path) => (
                    <li key={path}>GET {path}</li>
                  ))}
                </ul>
              </Panel>
            )}
          </aside>
        </div>
      </div>
    </>
  );
}

function registryAside(registry: CameraRegistry): string {
  switch (registry.state) {
    case 'unsupported':
      return 'Unavailable';
    case 'loading':
      return 'Loading';
    case 'error':
      return 'Error';
    case 'ready':
      return `${registry.cameras.length} registered`;
  }
}

function RegistryBody({ registry }: { registry: CameraRegistry }) {
  switch (registry.state) {
    case 'unsupported':
      return (
        <>
          <div className="camera-empty">
            <p className="camera-empty-title">No camera registry</p>
            <p className="empty">
              Backend camera discovery is not available yet, so there are no cameras to list. Each
              camera will appear as a card like the layout below.
            </p>
          </div>
          <ul className="camera-grid">
            <CameraCardPreview />
          </ul>
        </>
      );
    case 'loading':
      return <p className="empty">Loading cameras…</p>;
    case 'error':
      return <p className="inline-error">Cameras unavailable: {registry.error.message}</p>;
    case 'ready':
      return registry.cameras.length === 0 ? (
        <p className="empty">The backend reports no cameras.</p>
      ) : (
        <ul className="camera-grid">
          {registry.cameras.map((camera) => (
            <CameraCard key={camera.camera_id} camera={camera} />
          ))}
        </ul>
      );
  }
}
