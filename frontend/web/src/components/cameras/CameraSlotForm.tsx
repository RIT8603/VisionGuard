import type { CameraInfo, SourceType } from '../../api/types';
import type { CameraDraft } from '../../lib/cameraDrafts';
import { DASH } from '../../lib/format';
import { CameraStatusBadge } from '../pipeline/CameraStatusBadge';

interface CameraSlotFormProps {
  index: number;
  draft: CameraDraft;
  /** What the backend reports for this camera, if it is running. */
  live: CameraInfo | undefined;
  /** Shown under the fields; the page passes it only after an Apply attempt. */
  problem: string | null;
  disabled: boolean;
  canRemove: boolean;
  onChange: (patch: Partial<CameraDraft>) => void;
  onRemove: () => void;
}

const SOURCE_LABEL: Record<SourceType, string> = {
  rtsp: 'RTSP / IP camera',
  webcam: 'Webcam (USB or built-in)',
  file: 'Video file',
};

/** One camera slot: name, source type, and the field(s) that type needs. */
export function CameraSlotForm({
  index,
  draft,
  live,
  problem,
  disabled,
  canRemove,
  onChange,
  onRemove,
}: CameraSlotFormProps) {
  const base = `camera-${draft.key}`;
  const problemId = `${base}-problem`;

  return (
    <li className={problem ? 'camera-slot has-problem' : 'camera-slot'}>
      <header className="camera-slot-head">
        <h3>Camera {index + 1}</h3>
        {draft.id && <span className="mono camera-slot-id">{draft.id}</span>}
        <span className="spacer" />
        {live && <CameraStatusBadge status={live.status} />}
        <button
          type="button"
          className="camera-slot-remove"
          onClick={onRemove}
          disabled={disabled || !canRemove}
          aria-label={`Remove camera ${index + 1}`}
        >
          Remove
        </button>
      </header>

      <div className="camera-slot-body">
        <div className="field">
          <label htmlFor={`${base}-name`}>Name (optional)</label>
          <input
            id={`${base}-name`}
            type="text"
            value={draft.name}
            maxLength={60}
            placeholder="Main gate"
            disabled={disabled}
            onChange={(event) => onChange({ name: event.target.value })}
          />
        </div>

        <div className="field">
          <label htmlFor={`${base}-source`}>Source</label>
          <select
            id={`${base}-source`}
            value={draft.source}
            disabled={disabled}
            onChange={(event) => onChange({ source: event.target.value as SourceType })}
          >
            {(Object.keys(SOURCE_LABEL) as SourceType[]).map((source) => (
              <option key={source} value={source}>
                {SOURCE_LABEL[source]}
              </option>
            ))}
          </select>
        </div>

        {draft.source === 'rtsp' && (
          <div className="field">
            <label htmlFor={`${base}-url`}>Camera address</label>
            <input
              id={`${base}-url`}
              className="mono"
              type="text"
              value={draft.rtspUrl}
              placeholder="rtsp://user:password@192.168.1.10:554/stream1"
              autoComplete="off"
              spellCheck={false}
              disabled={disabled}
              aria-invalid={problem !== null}
              aria-describedby={problem ? problemId : undefined}
              onChange={(event) => onChange({ rtspUrl: event.target.value })}
            />
            <p className="field-help">
              The password is saved on the machine running VisionGuard and is never shown again
              (it appears as <code>***</code>). Leave that as it is to keep it.
            </p>
          </div>
        )}

        {draft.source === 'webcam' && (
          <div className="field">
            <label htmlFor={`${base}-device`}>Webcam number</label>
            <input
              id={`${base}-device`}
              className="mono"
              type="number"
              min={0}
              step={1}
              value={draft.deviceId}
              disabled={disabled}
              aria-invalid={problem !== null}
              aria-describedby={problem ? problemId : undefined}
              onChange={(event) => onChange({ deviceId: event.target.value })}
            />
            <p className="field-help">0 is the first camera on the machine running VisionGuard.</p>
          </div>
        )}

        {draft.source === 'file' && (
          <>
            <div className="field">
              <label htmlFor={`${base}-path`}>Video file path</label>
              <input
                id={`${base}-path`}
                className="mono"
                type="text"
                value={draft.path}
                placeholder="C:\Videos\campus.mp4"
                autoComplete="off"
                spellCheck={false}
                disabled={disabled}
                aria-invalid={problem !== null}
                aria-describedby={problem ? problemId : undefined}
                onChange={(event) => onChange({ path: event.target.value })}
              />
              <p className="field-help">A path on the machine running VisionGuard.</p>
            </div>
            <label className="field-check">
              <input
                type="checkbox"
                checked={draft.loop}
                disabled={disabled}
                onChange={(event) => onChange({ loop: event.target.checked })}
              />
              <span>Play again from the start when it ends</span>
            </label>
          </>
        )}

        {problem && (
          <p id={problemId} className="field-problem" role="alert">
            {problem}
          </p>
        )}

        {live && (
          <p className="camera-slot-live mono">
            {live.status === 'failed' && live.error
              ? live.error
              : `${live.fps === null ? DASH : live.fps.toFixed(1)} fps · ${
                  live.resolution ? `${live.resolution.width}×${live.resolution.height}` : DASH
                }`}
          </p>
        )}
      </div>
    </li>
  );
}
