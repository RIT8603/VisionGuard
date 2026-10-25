import { useState, type FormEvent } from 'react';
import { toApiError } from '../../api/client';
import { FACE_USER_FIELDS, type FaceUser, type FaceUserFields } from '../../api/faceTypes';
import { createFaceUser, deleteFaceUser, updateFaceUser } from '../../api/faces';
import { Panel } from '../ui/Panel';

const LABELS: Record<keyof FaceUserFields, string> = {
  name: 'Full name',
  employee_id: 'Employee / roll no.',
  role: 'Role',
  department: 'Department / class',
  notes: 'Notes',
};

const EMPTY: FaceUserFields = { name: '', employee_id: '', role: '', department: '', notes: '' };

function fieldsOf(user: FaceUser | null): FaceUserFields {
  if (!user) return EMPTY;
  return {
    name: user.name,
    employee_id: user.employee_id,
    role: user.role,
    department: user.department,
    notes: user.notes,
  };
}

interface Props {
  /** null = a new user. */
  user: FaceUser | null;
  editable: boolean;
  onSaved: (user: FaceUser, created: boolean) => void;
  onDeleted: (id: number) => void;
  onCancel: () => void;
}

/** Add or edit one user's details. Keyed by user id by the parent, so state resets per user. */
export function UserForm({ user, editable, onSaved, onDeleted, onCancel }: Props) {
  const [draft, setDraft] = useState<FaceUserFields>(() => fieldsOf(user));
  const [active, setActive] = useState(user?.active ?? true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isNew = user === null;
  const disabled = !editable || busy;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!draft.name.trim()) {
      setError('A name is required.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const saved = isNew ? await createFaceUser(draft) : await updateFaceUser(user.id, { ...draft, active });
      onSaved(saved, isNew);
    } catch (caught) {
      setError(toApiError(caught).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!user) return;
    if (!window.confirm(`Delete ${user.name}? This removes all of their face templates.`)) return;
    setBusy(true);
    setError(null);
    try {
      await deleteFaceUser(user.id);
      onDeleted(user.id);
    } catch (caught) {
      setError(toApiError(caught).message);
      setBusy(false);
    }
  };

  return (
    <Panel title={isNew ? 'Add user' : 'Details'}>
      <form className="face-form" onSubmit={submit}>
        {(Object.keys(FACE_USER_FIELDS) as (keyof FaceUserFields)[]).map((key) => {
          const id = `face-${key}`;
          const props = {
            id,
            value: draft[key],
            maxLength: FACE_USER_FIELDS[key],
            disabled,
            onChange: (event: { target: { value: string } }) => setDraft({ ...draft, [key]: event.target.value }),
          };
          return (
            <div key={key} className={key === 'notes' ? 'field face-field-wide' : 'field'}>
              <label htmlFor={id}>
                {LABELS[key]}
                {key === 'name' && ' (required, unique)'}
              </label>
              {key === 'notes' ? <textarea rows={2} {...props} /> : <input type="text" required={key === 'name'} {...props} />}
            </div>
          );
        })}

        {!isNew && (
          <label className="field-check face-field-wide">
            <input type="checkbox" checked={active} disabled={disabled} onChange={(e) => setActive(e.target.checked)} />
            Active - recognised on the cameras
          </label>
        )}

        {error && <p className="inline-error face-field-wide" role="alert">{error}</p>}

        {editable && (
          <div className="face-actions face-field-wide">
            <button type="submit" className="summary-btn" disabled={busy}>
              {isNew ? 'Save and enrol' : 'Save'}
            </button>
            <button type="button" className="summary-btn summary-btn-quiet" disabled={busy} onClick={onCancel}>
              {isNew ? 'Cancel' : 'Close'}
            </button>
            {!isNew && (
              <button type="button" className="summary-btn summary-btn-quiet face-danger" disabled={busy} onClick={remove}>
                Delete user
              </button>
            )}
          </div>
        )}
      </form>
    </Panel>
  );
}
