import type { CameraInfo, SourceType } from '../../api/types';
import {
  CATEGORIES,
  MODULE_CATALOG,
  backendKeysFor,
  isCoveredBy,
} from '../../content/moduleCatalog';
import type { CameraDraft } from '../../lib/cameraDrafts';
import { DASH } from '../../lib/format';
import { CameraStatusBadge } from '../pipeline/CameraStatusBadge';
import { VideoUploadField } from './VideoUploadField';

interface CameraSlotFormProps {
  index: number;
  draft: CameraDraft;
  /** What the backend reports for this camera, if it is running. */
  live: CameraInfo | undefined;
  /** Shown under the fields; the page passes it only after an Apply attempt. */
  problem: string | null;
  disabled: boolean;
  canRemove: boolean;
  /** Catalogue ids of the shared selection (Modules page), or null if unknown. */
  sharedModuleIds: readonly string[] | null;
  onChange: (patch: Partial<CameraDraft>) => void;
  onRemove: () => void;
}

const SOURCE_LABEL: Record<SourceType, string> = {
  rtsp: 'RTSP / IP camera',
  youtube: 'YouTube Stream',
  file: 'Video file',
  webcam: 'Webcam / HDMI capture card',
};

/** One camera slot: name, source type, and the field(s) that type needs. */
export function CameraSlotForm({
  index,
  draft,
  live,
  problem,
  disabled,
  canRemove,
  sharedModuleIds,
  onChange,
  onRemove,
}: CameraSlotFormProps) {
  const base = `camera-${draft.key}`;
  const problemId = `${base}-problem`;

  const chosen = new Set(draft.moduleIds);
  // A module whose backend parts are all switched on by other choices runs
  // anyway (e.g. people tracking under restricted zones), so show it as on.
  const chosenKeys = new Set(backendKeysFor(chosen));

  const setOwnModules = (own: boolean) => {
    // First switch to "own": start from what the camera runs today.
    if (own && draft.moduleIds.length === 0 && sharedModuleIds) {
      onChange({ ownModules: true, moduleIds: [...sharedModuleIds] });
    } else {
      onChange({ ownModules: own });
    }
  };

  const toggleModule = (id: string) =>
    onChange({
      moduleIds: chosen.has(id) ? draft.moduleIds.filter((m) => m !== id) : [...draft.moduleIds, id],
    });

  const sharedNames =
    sharedModuleIds === null
      ? null
      : MODULE_CATALOG.filter((m) => sharedModuleIds.includes(m.id)).map((m) => m.name);

  // Shown at the top of the card: below the module list it is off-screen.
  const failure = live?.status === 'failed' ? live.error : null;

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
        {failure && (
          <p className="camera-slot-live camera-slot-failure mono" role="status">
            {failure}
          </p>
        )}

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

        {draft.source === 'youtube' && (
          <div className="field">
            <label htmlFor={`${base}-youtube-url`}>YouTube stream URL</label>
            <input
              id={`${base}-youtube-url`}
              className="mono"
              type="text"
              value={draft.youtubeUrl}
              placeholder="https://www.youtube.com/watch?v=... or https://www.youtube.com/live/..."
              autoComplete="off"
              spellCheck={false}
              disabled={disabled}
              aria-invalid={problem !== null}
              aria-describedby={problem ? problemId : undefined}
              onChange={(event) => onChange({ youtubeUrl: event.target.value })}
            />
            <p className="field-help">
              Enter a public YouTube Live stream or video URL (e.g. <code>https://www.youtube.com/watch?v=...</code>, <code>https://www.youtube.com/live/...</code>, or <code>https://youtu.be/...</code>).
            </p>
          </div>
        )}

        {draft.source === 'webcam' && (
          <div className="field">
            <label htmlFor={`${base}-device`}>Device number</label>
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
            <p className="field-help">
              0 is the first camera on the machine running VisionGuard. An HDMI capture card is
              another camera device: usually 1 on a laptop with its own webcam, or 0 if it is the
              only one.
            </p>
          </div>
        )}

        {draft.source === 'file' && (
          <>
            <VideoUploadField
              id={base}
              disabled={disabled}
              path={draft.path}
              onPick={(path) => onChange({ path })}
            />
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
              <p className="field-help">
                Filled in by an upload, or type a path on the machine running VisionGuard.
              </p>
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

        <div className="field">
          <label htmlFor={`${base}-modules`}>AI modules</label>
          <select
            id={`${base}-modules`}
            value={draft.ownModules ? 'own' : 'shared'}
            disabled={disabled}
            onChange={(event) => setOwnModules(event.target.value === 'own')}
          >
            <option value="shared">Same as the Modules page</option>
            <option value="own">Choose for this camera</option>
          </select>
          {!draft.ownModules && (
            <p className="field-help">
              {sharedNames === null
                ? 'Runs whatever is selected on the Modules page.'
                : sharedNames.length === 0
                  ? 'Nothing is selected on the Modules page, so this camera only shows video.'
                  : `Currently: ${sharedNames.join(', ')}.`}
            </p>
          )}
        </div>

        {draft.ownModules && (
          <div className="camera-modules">
            {CATEGORIES.map((category) => (
              <fieldset key={category} className="camera-modules-group" disabled={disabled}>
                <legend>{category}</legend>
                {MODULE_CATALOG.filter(
                  (m) => m.category === category && m.status !== 'planned',
                ).map((module) => {
                  const picked = chosen.has(module.id);
                  const implied = !picked && isCoveredBy(module, chosenKeys);
                  return (
                    <label key={module.id} className="field-check" title={module.summary}>
                      <input
                        type="checkbox"
                        checked={picked || implied}
                        disabled={implied}
                        onChange={() => toggleModule(module.id)}
                      />
                      <span>
                        {module.name}
                        {implied && <em className="camera-modules-tag"> included</em>}
                        {module.status === 'setup' && (
                          <em className="camera-modules-tag"> needs setup</em>
                        )}
                      </span>
                    </label>
                  );
                })}
              </fieldset>
            ))}
            <p className="field-help">
              {draft.moduleIds.length === 0
                ? 'Nothing chosen: this camera will only show video.'
                : 'This camera runs only these, whatever the Modules page says.'}
            </p>
          </div>
        )}

        {problem && (
          <p id={problemId} className="field-problem" role="alert">
            {problem}
          </p>
        )}

        {live && !failure && (
          <p className="camera-slot-live mono">
            {`${live.fps === null ? DASH : live.fps.toFixed(1)} fps · ${
              live.resolution ? `${live.resolution.width}×${live.resolution.height}` : DASH
            }`}
          </p>
        )}
      </div>
    </li>
  );
}
