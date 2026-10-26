import { useState } from 'react';
import { Link } from 'react-router';
import { toApiError } from '../../api/client';
import { createFaceUser, deleteFaceUser, setModules, updateFaceUser } from '../../api/endpoints';
import type { FaceStatus, FaceUser, FaceUserInput } from '../../api/types';
import { PATHS } from '../../app/paths';
import { EnrollmentWizard } from '../../components/faces/EnrollmentWizard';
import { RecognitionCheck } from '../../components/faces/RecognitionCheck';
import { UserForm } from '../../components/faces/UserForm';
import { useFaceRegistry } from '../../hooks/useFaceRegistry';
import { usePipeline } from '../../state/PipelineProvider';

type Panel = { kind: 'add' } | { kind: 'edit'; id: number } | { kind: 'enroll'; id: number } | null;

export function FacesPage() {
  const registry = useFaceRegistry();
  const { status: pipeline } = usePipeline();
  const [panel, setPanel] = useState<Panel>(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [enabling, setEnabling] = useState<'idle' | 'applying' | 'error'>('idle');
  const [enableError, setEnableError] = useState<string | null>(null);

  const faceStatus = registry.status;
  const editable = faceStatus?.editable ?? false;

  // Are the two modules that act on this registry actually running?
  const enabledModules = new Set(pipeline.data?.info.modules.map((m) => m.name) ?? []);
  const detectionOn = enabledModules.has('face_detection');
  const recognitionOn = enabledModules.has('face_recognition');
  const pipelineKnown = pipeline.data !== null;

  // Turn the two modules on without leaving the page. The backend rebuilds the
  // pipeline in the background; the banner disappears once status reports them.
  const enableRecognition = async () => {
    setEnabling('applying');
    setEnableError(null);
    try {
      const wanted = new Set(enabledModules);
      wanted.add('face_detection');
      wanted.add('face_recognition');
      await setModules([...wanted]);
    } catch (caught) {
      setEnabling('error');
      setEnableError(toApiError(caught).message);
    }
  };

  const query = filter.trim().toLowerCase();
  const users = registry.users.filter(
    (user) =>
      query === '' ||
      [user.name, user.employee_id, user.role, user.department].some((v) => v.toLowerCase().includes(query)),
  );

  const open = (next: Panel) => {
    setFormError(null);
    setActionError(null);
    setPanel(next);
  };

  const submitUser = async (input: FaceUserInput) => {
    setBusy(true);
    setFormError(null);
    try {
      if (panel?.kind === 'edit') {
        registry.replaceUser(await updateFaceUser(panel.id, input));
        setPanel(null);
      } else {
        const created = await createFaceUser(input);
        registry.replaceUser(created);
        // A new user has no face yet: go straight to capturing it.
        setPanel({ kind: 'enroll', id: created.id });
      }
    } catch (caught) {
      setFormError(toApiError(caught).message);
    } finally {
      setBusy(false);
    }
  };

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setActionError(null);
    try {
      await action();
    } catch (caught) {
      setActionError(toApiError(caught).message);
    } finally {
      setBusy(false);
    }
  };

  const toggleActive = (user: FaceUser) =>
    run(async () => registry.replaceUser(await updateFaceUser(user.id, { active: !user.active })));

  const remove = (user: FaceUser) =>
    run(async () => {
      const sure = window.confirm(
        `Delete ${user.name}? Their details and every stored face template are removed for good, and they will no longer be recognised.`,
      );
      if (!sure) return;
      await deleteFaceUser(user.id);
      registry.removeUser(user.id);
      if (panel && 'id' in panel && panel.id === user.id) setPanel(null);
    });

  const selected =
    panel && 'id' in panel ? (registry.users.find((user) => user.id === panel.id) ?? null) : null;

  return (
    <>
      <title>Face Registry · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Face Registry</h2>
          <p className="page-intro">
            Add the people VisionGuard should recognise, then capture each person's face from several
            angles. Registered people are named on the live video; everyone else is shown as unknown. Only
            numeric face templates are stored, never photographs.
          </p>
        </header>

        {registry.error && (
          <p className="inline-error">Face registry unavailable: {registry.error.message}</p>
        )}

        {faceStatus && <StatusStrip status={faceStatus} />}

        {faceStatus && !faceStatus.models.ready && (
          <div className="notice notice-warn faces-notice" role="alert">
            <span className="notice-word">Models</span>
            <div className="notice-body">
              <strong>Face models are not installed.</strong> Capturing and recognising faces needs two
              small model files. Run this once in the project folder, then reload this page:
              <pre className="notice-code mono">python -m tools.download_face_models</pre>
            </div>
          </div>
        )}

        {pipelineKnown && (!detectionOn || !recognitionOn) && (
          <div className="notice notice-info faces-notice" role="status">
            <span className="notice-word">Live off</span>
            <div className="notice-body">
              Live recognition is <strong>off</strong>: {!detectionOn && 'face_detection'}
              {!detectionOn && !recognitionOn && ' and '}
              {!recognitionOn && 'face_recognition'} {detectionOn !== recognitionOn ? 'is' : 'are'} not
              enabled. Users can still be registered now.{' '}
              <Link to={PATHS.portal.modules}>Modules page</Link> has the full list.
              <div className="faces-notice-actions">
                <button
                  type="button"
                  className="summary-btn"
                  disabled={enabling === 'applying' || !pipelineKnown}
                  onClick={() => void enableRecognition()}
                >
                  {enabling === 'applying' ? 'Applying…' : 'Enable live recognition'}
                </button>
                {enabling === 'applying' && (
                  <span className="field-help">
                    The pipeline is restarting with face recognition; this takes a few seconds.
                  </span>
                )}
                {enableError && <span className="field-problem">{enableError}</span>}
              </div>
            </div>
          </div>
        )}

        {faceStatus && !faceStatus.editable && (
          <p className="inline-error">
            The registry is read-only (dashboard.allow_face_edit is off), so users cannot be added or
            changed.
          </p>
        )}

        <div className="faces-toolbar">
          <input
            type="search"
            className="faces-search"
            placeholder="Search name, ID, role…"
            aria-label="Search users"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
          <span className="page-meta mono">
            {users.length} of {registry.users.length} users
          </span>
          <button
            type="button"
            className="summary-btn"
            disabled={!editable || busy}
            onClick={() => open({ kind: 'add' })}
          >
            Add user
          </button>
        </div>

        {actionError && <p className="inline-error">{actionError}</p>}

        {panel?.kind === 'add' && (
          <section className="panel faces-form-panel">
            <header className="panel-head">
              <h2>New user</h2>
            </header>
            <div className="panel-body">
              <UserForm busy={busy} error={formError} onSubmit={(v) => void submitUser(v)} onCancel={() => setPanel(null)} />
            </div>
          </section>
        )}

        {panel?.kind === 'edit' && selected && (
          <section className="panel faces-form-panel">
            <header className="panel-head">
              <h2>Edit · {selected.name}</h2>
            </header>
            <div className="panel-body">
              <UserForm
                user={selected}
                busy={busy}
                error={formError}
                onSubmit={(v) => void submitUser(v)}
                onCancel={() => setPanel(null)}
              />
            </div>
          </section>
        )}

        {panel?.kind === 'enroll' && selected && faceStatus && (
          <EnrollmentWizard
            key={selected.id}
            user={selected}
            status={faceStatus}
            onUserChange={registry.replaceUser}
            onClose={() => setPanel(null)}
          />
        )}

        {users.length === 0 ? (
          <p className="empty">
            {registry.loading
              ? 'Loading users…'
              : registry.users.length === 0
                ? 'No users yet. Press “Add user” to register the first person.'
                : 'No users match that search.'}
          </p>
        ) : (
          <div className="faces-table-wrap">
            <table className="faces-table">
              <thead>
                <tr>
                  <th scope="col">Name</th>
                  <th scope="col">ID</th>
                  <th scope="col">Role</th>
                  <th scope="col">Face data</th>
                  <th scope="col">Status</th>
                  <th scope="col">
                    <span className="visually-hidden">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id} className={user.active ? '' : 'is-inactive'}>
                    <th scope="row">
                      <span className="faces-name">{user.name}</span>
                      {user.department && <span className="faces-sub">{user.department}</span>}
                    </th>
                    <td className="mono">{user.employee_id || '—'}</td>
                    <td>{user.role || '—'}</td>
                    <td>
                      <PoseChips user={user} status={faceStatus} />
                    </td>
                    <td>
                      <span className={`faces-badge ${badgeClass(user)}`}>{badgeText(user)}</span>
                    </td>
                    <td className="faces-actions">
                      <button
                        type="button"
                        className="link-btn"
                        disabled={!editable || busy}
                        onClick={() => open({ kind: 'enroll', id: user.id })}
                      >
                        {user.sample_count === 0 ? 'Capture face' : 'Add / retake'}
                      </button>
                      <button
                        type="button"
                        className="link-btn"
                        disabled={!editable || busy}
                        onClick={() => open({ kind: 'edit', id: user.id })}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        className="link-btn"
                        disabled={!editable || busy}
                        onClick={() => void toggleActive(user)}
                      >
                        {user.active ? 'Deactivate' : 'Activate'}
                      </button>
                      <button
                        type="button"
                        className="link-btn link-btn-danger"
                        disabled={!editable || busy}
                        onClick={() => void remove(user)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {faceStatus?.models.ready && registry.users.length > 0 && <RecognitionCheck />}
      </div>
    </>
  );
}

function badgeText(user: FaceUser): string {
  if (!user.active) return 'Inactive';
  if (user.enrolled) return 'Enrolled';
  if (user.sample_count > 0) return 'Incomplete';
  return 'No face yet';
}

function badgeClass(user: FaceUser): string {
  if (!user.active) return 'is-off';
  if (user.enrolled) return 'is-ok';
  return 'is-warn';
}

function PoseChips({ user, status }: { user: FaceUser; status: FaceStatus | null }) {
  const poses = status?.poses ?? [];
  if (poses.length === 0) return <span>{user.sample_count}</span>;
  return (
    <span className="pose-chips">
      {poses.map((pose) => {
        const count = user.poses[pose.id] ?? 0;
        return (
          <span
            key={pose.id}
            className={`pose-chip${count > 0 ? ' is-on' : ''}`}
            title={`${pose.id}: ${count} captured${pose.required ? ' (required)' : ''}`}
          >
            {pose.id[0].toUpperCase()}
          </span>
        );
      })}
    </span>
  );
}

function StatusStrip({ status }: { status: FaceStatus }) {
  const items: [string, string][] = [
    ['Users', String(status.users)],
    ['Enrolled', String(status.enrolled)],
    ['Face templates', String(status.templates)],
    ['Models', status.models.ready ? 'ready' : 'missing'],
  ];
  return (
    <dl className="faces-strip">
      {items.map(([label, value]) => (
        <div key={label} className="faces-strip-item">
          <dt>{label}</dt>
          <dd className="mono">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
