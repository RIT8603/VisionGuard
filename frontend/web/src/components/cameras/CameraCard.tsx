import { Link } from 'react-router';
import { CAMERA_ENDPOINTS, type CameraStatus, type CameraSummary } from '../../api/cameras';
import { cameraPath } from '../../app/paths';
import { DASH, fmtTime } from '../../lib/format';
import type { ViewState } from '../../state/viewState';
import { StateBadge } from '../pipeline/StateBadge';

/** Camera states that mean the same as a pipeline ViewState reuse its badge. */
const STATUS_VIEW_STATE: Partial<Record<CameraStatus, ViewState>> = {
  online: 'live',
  starting: 'waiting',
  stale: 'stale',
  offline: 'offline',
  stopped: 'stopped',
};

function CameraStatusBadge({ status }: { status: CameraStatus }) {
  const viewState = STATUS_VIEW_STATE[status];
  // A state the backend added later: show its raw text rather than guess.
  return viewState ? <StateBadge state={viewState} /> : <span className="camera-status">{status}</span>;
}

/**
 * One registry entry: snapshot, identity, source, status and last frame.
 * Rendered only from real GET /api/cameras data, so the snapshot URL is
 * camera-scoped and never the shared /stream.mjpg.
 */
export function CameraCard({ camera }: { camera: CameraSummary }) {
  return (
    <li className="camera-card">
      <Link className="camera-card-link" to={cameraPath(camera.camera_id)}>
        <div className="camera-preview">
          <img
            src={CAMERA_ENDPOINTS.snapshot(camera.camera_id)}
            alt={`Latest snapshot from ${camera.name}`}
            loading="lazy"
          />
        </div>
        <div className="camera-card-body">
          <div className="camera-card-top">
            <h3 className="camera-card-name">{camera.name}</h3>
            <CameraStatusBadge status={camera.status} />
          </div>
          <CameraMeta
            id={camera.camera_id}
            source={camera.source_type}
            lastFrame={camera.last_frame_at ? fmtTime(camera.last_frame_at) : DASH}
          />
        </div>
      </Link>
    </li>
  );
}

/**
 * The card's layout with labels only - no name, id or status. Shown while no
 * registry exists so the design is reviewable without inventing a camera.
 */
export function CameraCardPreview() {
  return (
    <li className="camera-card camera-card-preview" aria-label="Camera card layout (no camera data)">
      <div className="camera-preview">
        <span className="camera-preview-empty">Snapshot</span>
      </div>
      <div className="camera-card-body">
        <div className="camera-card-top">
          <span className="camera-card-name camera-card-placeholder">Camera name</span>
          <span className="camera-status camera-status-placeholder">Status</span>
        </div>
        <CameraMeta id={DASH} source={DASH} lastFrame={DASH} />
        <p className="camera-card-note">Layout preview · not a camera</p>
      </div>
    </li>
  );
}

function CameraMeta({ id, source, lastFrame }: { id: string; source: string; lastFrame: string }) {
  const rows: [label: string, value: string][] = [
    ['Camera ID', id],
    ['Source', source],
    ['Last frame', lastFrame],
  ];

  return (
    <dl className="camera-card-meta">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd className="mono">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
