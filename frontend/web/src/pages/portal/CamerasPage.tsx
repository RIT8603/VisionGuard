import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import { PATHS } from '../../app/paths';
import { ApiError } from '../../api/client';
import { setCameras } from '../../api/endpoints';
import { CameraSlotForm } from '../../components/cameras/CameraSlotForm';
import { catalogIdsFor } from '../../content/moduleCatalog';
import { usePipeline } from '../../state/PipelineProvider';
import {
  blankDraft,
  draftFromCamera,
  signature,
  toInput,
  validateDrafts,
  type CameraDraft,
} from '../../lib/cameraDrafts';

type ApplyState = 'idle' | 'applying' | 'applied' | 'error';

export function CamerasPage() {
  const { cameras } = usePipeline();
  const maxCameras = cameras.data?.max_cameras ?? 8;
  const editable = cameras.data?.editable ?? false;

  const [drafts, setDrafts] = useState<CameraDraft[]>([]);
  const [countText, setCountText] = useState('');
  // What the backend was last told (or started with), to know if anything changed.
  const [applied, setApplied] = useState('');
  const [apply, setApply] = useState<{ state: ApplyState; message?: string }>({ state: 'idle' });
  const [showProblems, setShowProblems] = useState(false);
  const seeded = useRef(false);

  // Fill the form from the cameras the backend is running, once. After that
  // the operator's edits are the source of truth, not whatever is running.
  useEffect(() => {
    if (seeded.current || cameras.data === null) return;
    seeded.current = true;
    const initial = cameras.data.cameras.map(draftFromCamera);
    const start = initial.length > 0 ? initial : [blankDraft()];
    setDrafts(start);
    setCountText(String(start.length));
    setApplied(signature(initial));
  }, [cameras.data]);

  const problems = validateDrafts(drafts);
  const hasProblems = problems.some((problem) => problem !== null);
  const dirty = seeded.current && signature(drafts) !== applied;
  const busy = apply.state === 'applying';
  // Editing depends only on the backend's own switch (allow_camera_edit),
  // never on whether any camera is live: a camera that is down because its
  // address is wrong can only be fixed from this form.
  const loaded = cameras.data !== null;
  const locked = loaded && !editable;
  // The camera list itself is not answering - the backend process is away.
  const unreachable = cameras.error !== null;
  // Any camera that is down can be retried by re-applying, even unchanged.
  const anyDown =
    cameras.data?.cameras.some((c) => c.status === 'offline' || c.status === 'failed') ?? false;

  const parsedCount = Number(countText);
  const countValid =
    countText.trim() !== '' &&
    Number.isInteger(parsedCount) &&
    parsedCount >= 1 &&
    parsedCount <= maxCameras;

  // More slots appear as you type. Fewer wait until you commit (Enter or
  // click away): typing "12" passes through "1", and that must not delete
  // the slots you have already filled in.
  const onCount = (text: string) => {
    setCountText(text);
    const count = Number(text);
    if (text.trim() === '' || !Number.isInteger(count) || count < 1 || count > maxCameras) return;
    if (count > drafts.length) {
      setDrafts((previous) => [
        ...previous,
        ...Array.from({ length: count - previous.length }, blankDraft),
      ]);
    }
  };

  const commitCount = () => {
    if (countValid && parsedCount < drafts.length) {
      setDrafts((previous) => previous.slice(0, parsedCount));
      setCountText(String(parsedCount));
    } else {
      setCountText(String(drafts.length));
    }
  };

  const pendingRemoval = countValid && parsedCount < drafts.length;

  const patch = (key: string, change: Partial<CameraDraft>) =>
    setDrafts((previous) => previous.map((d) => (d.key === key ? { ...d, ...change } : d)));

  const remove = (key: string) => {
    const next = drafts.filter((d) => d.key !== key);
    setDrafts(next);
    setCountText(String(next.length));
  };

  const onApply = async () => {
    setShowProblems(true);
    if (hasProblems || drafts.length === 0) return;
    setApply({ state: 'applying' });
    try {
      const response = await setCameras(drafts.map(toInput));
      // The backend numbers new cameras; remember those ids so the next apply
      // keeps each camera's history instead of renumbering.
      const withIds = drafts.map((draft, index) => ({
        ...draft,
        id: response.cameras[index]?.id ?? draft.id,
      }));
      setDrafts(withIds);
      setApplied(signature(withIds));
      setApply({ state: 'applied' });
    } catch (error: unknown) {
      const message = error instanceof ApiError ? error.message : 'Could not reach the backend';
      setApply({ state: 'error', message });
    }
  };

  const liveFor = (draft: CameraDraft) =>
    draft.id ? cameras.data?.cameras.find((c) => c.id === draft.id) : undefined;

  const applyError = cameras.data?.apply_error ?? null;
  const sharedKeys = cameras.data?.shared_modules;
  const sharedModuleIds = Array.isArray(sharedKeys) ? catalogIdsFor(sharedKeys) : null;

  return (
    <>
      <title>Cameras · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Cameras</h2>
          <p className="page-intro">
            Say how many cameras you have, then fill in each one. Every camera is analysed on its
            own, with the modules chosen on the Modules page or with its own set picked here, so
            each camera can do a different job. The AI models are shared, so more cameras means a
            slower frame rate on each, not more memory.
          </p>
        </header>

        {unreachable && (
          <div className="notice notice-crit" role="alert">
            <span className="notice-word">Offline</span>
            <div className="notice-body">
              The backend is not answering. You can keep editing; Apply will go through once it
              is running again.
            </div>
          </div>
        )}
        {!unreachable && locked && (
          <div className="notice notice-warn" role="note">
            <span className="notice-word">Locked</span>
            <div className="notice-body">
              This backend does not allow camera changes from the dashboard
              (<code>dashboard.allow_camera_edit</code>). Edit <code>configs/cameras.yaml</code> and
              restart instead.
            </div>
          </div>
        )}

        {apply.state === 'applying' && (
          <div className="notice notice-info" role="status">
            <span className="notice-word">Applying</span>
            <div className="notice-body">
              Sending the camera list. Every camera restarts, so the video pauses for a few seconds.
            </div>
          </div>
        )}
        {apply.state === 'applied' && !applyError && (
          <div className="notice notice-info" role="status">
            <span className="notice-word">Applied</span>
            <div className="notice-body">
              The cameras are restarting and the list was saved to{' '}
              <code>configs/cameras.yaml</code>. Watch each one come up below, or{' '}
              <Link to={PATHS.portal.dashboard}>open the dashboard</Link>.
            </div>
          </div>
        )}
        {apply.state === 'error' && (
          <div className="notice notice-warn" role="alert">
            <span className="notice-word">Failed</span>
            <div className="notice-body">
              Could not apply the cameras: {apply.message}. Nothing was changed.
            </div>
          </div>
        )}
        {applyError && (
          <div className="notice notice-warn" role="alert">
            <span className="notice-word">Not applied</span>
            <div className="notice-body">
              The backend could not start the last change and kept the previous setup: {applyError}
            </div>
          </div>
        )}

        <div className="camera-toolbar">
          <div className="field camera-count">
            <label htmlFor="camera-count">How many cameras?</label>
            <input
              id="camera-count"
              className="mono"
              type="number"
              inputMode="numeric"
              min={1}
              max={maxCameras}
              step={1}
              value={countText}
              disabled={locked || busy}
              aria-invalid={countText !== '' && !countValid}
              aria-describedby="camera-count-help"
              onChange={(event) => onCount(event.target.value)}
              onBlur={commitCount}
              onKeyDown={(event) => {
                if (event.key === 'Enter') commitCount();
              }}
            />
          </div>
          <p
            id="camera-count-help"
            className={countText !== '' && !countValid ? 'field-problem' : 'field-help'}
          >
            {countText !== '' && !countValid
              ? `Enter a whole number from 1 to ${maxCameras}.`
              : pendingRemoval
                ? `Camera ${parsedCount + 1}${
                    drafts.length > parsedCount + 1 ? `–${drafts.length}` : ''
                  } will be removed when you press Enter or click away.`
                : `Up to ${maxCameras}. Raising the number adds empty slots.`}
          </p>
          <span className="spacer" />
          {dirty && apply.state !== 'applying' && (
            <span className="camera-dirty">Unsaved changes</span>
          )}
          <button
            type="button"
            className="summary-btn"
            onClick={onApply}
            disabled={locked || busy || !loaded || (!dirty && !anyDown)}
          >
            {busy ? 'Applying…' : !dirty && anyDown ? 'Retry cameras' : 'Apply cameras'}
          </button>
        </div>

        <ol className="camera-slots">
          {drafts.map((draft, index) => (
            <CameraSlotForm
              key={draft.key}
              index={index}
              draft={draft}
              live={liveFor(draft)}
              problem={showProblems ? problems[index] : null}
              disabled={locked || busy}
              canRemove={drafts.length > 1}
              sharedModuleIds={sharedModuleIds}
              onChange={(change) => patch(draft.key, change)}
              onRemove={() => remove(draft.key)}
            />
          ))}
        </ol>

        {!loaded && !unreachable && <p className="empty">Loading the camera list…</p>}
      </div>
    </>
  );
}
