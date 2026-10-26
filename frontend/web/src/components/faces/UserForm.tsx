import { useState, type FormEvent } from 'react';
import type { FaceUser, FaceUserInput } from '../../api/types';

interface UserFormProps {
  /** The user being edited; omit to add a new one. */
  user?: FaceUser;
  busy: boolean;
  error: string | null;
  onSubmit: (input: FaceUserInput) => void;
  onCancel: () => void;
}

const LIMITS = { name: 80, employee_id: 40, role: 60, department: 60, notes: 500 } as const;

export function UserForm({ user, busy, error, onSubmit, onCancel }: UserFormProps) {
  const [values, setValues] = useState<FaceUserInput>({
    name: user?.name ?? '',
    employee_id: user?.employee_id ?? '',
    role: user?.role ?? '',
    department: user?.department ?? '',
    notes: user?.notes ?? '',
  });
  const [touched, setTouched] = useState(false);

  const set = (key: keyof FaceUserInput) => (value: string) =>
    setValues((previous) => ({ ...previous, [key]: value }));

  const nameProblem = values.name.trim() === '' ? 'A name is required.' : null;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setTouched(true);
    if (nameProblem) return;
    onSubmit(values);
  };

  const field = (key: keyof FaceUserInput, label: string, hint?: string) => (
    <div className="field">
      <label htmlFor={`user-${key}`}>{label}</label>
      <input
        id={`user-${key}`}
        type="text"
        value={values[key]}
        maxLength={LIMITS[key]}
        disabled={busy}
        aria-invalid={key === 'name' && touched && nameProblem !== null}
        onChange={(event) => set(key)(event.target.value)}
      />
      {key === 'name' && touched && nameProblem ? (
        <p className="field-problem">{nameProblem}</p>
      ) : (
        hint && <p className="field-help">{hint}</p>
      )}
    </div>
  );

  return (
    <form className="user-form" onSubmit={submit} noValidate>
      <div className="user-form-grid">
        {field('name', 'Full name', 'Shown on the live video when this person is recognised.')}
        {field('employee_id', 'ID / Roll no.')}
        {field('role', 'Role (student, staff...)')}
        {field('department', 'Class / Department')}
      </div>
      <div className="field">
        <label htmlFor="user-notes">Notes</label>
        <input
          id="user-notes"
          type="text"
          value={values.notes}
          maxLength={LIMITS.notes}
          disabled={busy}
          onChange={(event) => set('notes')(event.target.value)}
        />
      </div>
      {error && <p className="inline-error">{error}</p>}
      <div className="user-form-actions">
        <button type="submit" className="summary-btn" disabled={busy}>
          {busy ? 'Saving…' : user ? 'Save changes' : 'Add user'}
        </button>
        <button type="button" className="summary-btn summary-btn-quiet user-form-cancel" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
