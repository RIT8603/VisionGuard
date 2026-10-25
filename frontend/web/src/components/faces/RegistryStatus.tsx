import type { FaceStatus } from '../../api/faceTypes';
import { Panel } from '../ui/Panel';

/** Model readiness and registry totals. */
export function RegistryStatus({ status }: { status: FaceStatus }) {
  const { models } = status;
  return (
    <Panel
      title="Registry"
      aside={<span className={models.ready ? 'face-ok' : 'face-warn'}>{models.ready ? 'Models: ready' : 'Models: missing'}</span>}
    >
      <dl className="face-stats">
        <div>
          <dt>Users</dt>
          <dd className="mono">{status.users}</dd>
        </div>
        <div>
          <dt>Enrolled</dt>
          <dd className="mono">{status.enrolled}</dd>
        </div>
        <div>
          <dt>Templates</dt>
          <dd className="mono">{status.templates}</dd>
        </div>
        <div>
          <dt>Match threshold</dt>
          <dd className="mono">{status.threshold.toFixed(2)}</dd>
        </div>
      </dl>

      {!models.ready && (
        <div className="face-callout">
          <p>
            The face models are not installed, so enrolment and testing are off. From the
            repository root, run once:
          </p>
          <code>python -m tools.download_face_models</code>
          <ul className="face-files mono">
            <li>{models.detection.present ? '✓' : '✗'} {models.detection.path}</li>
            <li>{models.recognition.present ? '✓' : '✗'} {models.recognition.path}</li>
          </ul>
        </div>
      )}
      {models.error && <p className="inline-error">Model error: {models.error}</p>}
    </Panel>
  );
}
