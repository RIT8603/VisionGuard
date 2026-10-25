import type { FacePoseInfo, FaceUser } from '../../api/faceTypes';
import { Panel } from '../ui/Panel';

interface Props {
  users: FaceUser[];
  poses: FacePoseInfo[];
  selectedId: number | null;
  editable: boolean;
  onSelect: (user: FaceUser) => void;
  onAdd: () => void;
}

function Badge({ user }: { user: FaceUser }) {
  if (!user.active) return <span className="face-badge face-badge-off">Inactive</span>;
  if (user.enrolled) return <span className="face-badge face-badge-ok">Enrolled</span>;
  return <span className="face-badge face-badge-warn">Needs {user.missing_poses.join(', ')}</span>;
}

/** Everyone in the registry. Selecting a user opens their details and enrolment. */
export function UserList({ users, poses, selectedId, editable, onSelect, onAdd }: Props) {
  return (
    <Panel title="Users" aside={`${users.length}`}>
      {editable && (
        <button type="button" className="summary-btn face-add" onClick={onAdd}>
          Add user
        </button>
      )}
      {users.length === 0 ? (
        <p className="empty">No one is registered yet.</p>
      ) : (
        <ul className="face-users">
          {users.map((user) => (
            <li key={user.id}>
              <button
                type="button"
                className="face-user"
                aria-current={user.id === selectedId ? 'true' : undefined}
                onClick={() => onSelect(user)}
              >
                <span className="face-user-name">{user.name}</span>
                <span className="face-user-meta muted">
                  {[user.employee_id, user.role, user.department].filter(Boolean).join(' · ') || '—'}
                </span>
                <span className="face-user-poses mono">
                  {poses.map((pose) => (
                    <span key={pose.id} className={(user.poses[pose.id] ?? 0) > 0 ? 'face-pose-on' : undefined}>
                      {pose.id[0].toUpperCase()}
                    </span>
                  ))}
                </span>
                <Badge user={user} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
