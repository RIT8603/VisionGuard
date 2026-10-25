import { useCallback, useEffect, useState } from 'react';
import { toApiError } from '../../api/client';
import type { FaceStatus, FaceUser } from '../../api/faceTypes';
import { getFaceStatus, listFaceUsers } from '../../api/faces';
import { EnrollmentWizard } from '../../components/faces/EnrollmentWizard';
import { RecognitionCheck } from '../../components/faces/RecognitionCheck';
import { RegistryStatus } from '../../components/faces/RegistryStatus';
import { UserForm } from '../../components/faces/UserForm';
import { UserList } from '../../components/faces/UserList';
import { usePageVisible } from '../../hooks/usePageVisible';

/** null = nothing open; 'new' = the add-user form; a number = that user. */
type Selection = null | 'new' | number;

export function FacesPage() {
  const visible = usePageVisible();
  const [status, setStatus] = useState<FaceStatus | null>(null);
  const [users, setUsers] = useState<FaceUser[]>([]);
  const [loadError, setLoadError] = useState<{ unavailable: boolean; message: string } | null>(null);
  const [selection, setSelection] = useState<Selection>(null);

  const load = useCallback(async () => {
    try {
      const [nextStatus, nextUsers] = await Promise.all([getFaceStatus(), listFaceUsers()]);
      setStatus(nextStatus);
      setUsers(nextUsers);
      setLoadError(null);
    } catch (caught) {
      const error = toApiError(caught);
      setLoadError({ unavailable: error.status === 503, message: error.message });
    }
  }, []);

  useEffect(() => {
    if (visible) void load();
  }, [visible, load]);

  /** A user changed (details or templates): update the list and the totals. */
  const userChanged = useCallback(
    (user: FaceUser) => {
      setUsers((current) => {
        const exists = current.some((entry) => entry.id === user.id);
        return exists ? current.map((entry) => (entry.id === user.id ? user : entry)) : [...current, user];
      });
      void getFaceStatus().then(setStatus, () => undefined);
    },
    [],
  );

  const selected = typeof selection === 'number' ? users.find((user) => user.id === selection) ?? null : null;
  const editable = status?.editable ?? false;

  return (
    <>
      <title>Face Registry · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          {/* h2: the portal header's brand is the page's h1. */}
          <h2 className="page-title">Face Registry</h2>
          <p className="page-intro">
            Add the people VisionGuard should recognise and capture each face from several angles. A registered person
            is named on the video from the next frame - no restart. Only face templates (numbers) are stored, never
            photos. Enable face detection and face recognition on the Modules page.
          </p>
        </header>

        {loadError?.unavailable && (
          <div className="notice notice-warn" role="status">
            <span className="notice-word">Unavailable</span>
            <div className="notice-body">{loadError.message}</div>
          </div>
        )}
        {loadError && !loadError.unavailable && (
          <p className="inline-error">Face Registry unavailable: {loadError.message}</p>
        )}
        {status && !editable && (
          <p className="page-meta mono">Read-only for this run (dashboard.allow_face_edit: false).</p>
        )}

        {status && (
          <div className="face-layout">
            <div className="face-col">
              <RegistryStatus status={status} />
              <UserList
                users={users}
                poses={status.poses}
                selectedId={typeof selection === 'number' ? selection : null}
                editable={editable}
                onSelect={(user) => setSelection(user.id)}
                onAdd={() => setSelection('new')}
              />
            </div>

            <div className="face-col">
              {selection === 'new' && (
                <UserForm
                  key="new"
                  user={null}
                  editable={editable}
                  onSaved={(user) => {
                    userChanged(user);
                    setSelection(user.id);
                  }}
                  onDeleted={() => undefined}
                  onCancel={() => setSelection(null)}
                />
              )}
              {selected && (
                <>
                  <UserForm
                    key={selected.id}
                    user={selected}
                    editable={editable}
                    onSaved={userChanged}
                    onDeleted={(id) => {
                      setUsers((current) => current.filter((user) => user.id !== id));
                      setSelection(null);
                      void getFaceStatus().then(setStatus, () => undefined);
                    }}
                    onCancel={() => setSelection(null)}
                  />
                  <EnrollmentWizard key={`enrol-${selected.id}`} user={selected} status={status} onUserChange={userChanged} />
                </>
              )}
              {selection === null && (
                <p className="empty face-hint">
                  {users.length ? 'Select a user to edit them or capture their face.' : editable ? 'Add a user to start.' : 'No users yet.'}
                </p>
              )}
              <RecognitionCheck modelsReady={status.models.ready} />
            </div>
          </div>
        )}
      </div>
    </>
  );
}
