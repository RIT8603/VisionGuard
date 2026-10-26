import { useCallback, useEffect, useState } from 'react';
import { ApiError, toApiError } from '../api/client';
import { getFaceStatus, getFaceUsers } from '../api/endpoints';
import type { FaceStatus, FaceUser } from '../api/types';

export interface FaceRegistry {
  status: FaceStatus | null;
  users: FaceUser[];
  error: ApiError | null;
  loading: boolean;
  /** Reload the status and the user list from the backend. */
  refresh: () => void;
  /** Put an updated user (from a mutation's response) into the list without a round trip. */
  replaceUser: (user: FaceUser) => void;
  removeUser: (id: number) => void;
}

/**
 * Loads the face registry once on mount, then only when asked.
 *
 * Not polled: the registry changes only when this operator changes it, and
 * every mutation already returns the new state of what it touched.
 */
export function useFaceRegistry(): FaceRegistry {
  const [status, setStatus] = useState<FaceStatus | null>(null);
  const [users, setUsers] = useState<FaceUser[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    Promise.all([getFaceStatus(controller.signal), getFaceUsers(controller.signal)])
      .then(([nextStatus, nextUsers]) => {
        setStatus(nextStatus);
        setUsers(nextUsers.users);
        setError(null);
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;
        setError(toApiError(caught));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  const replaceUser = useCallback((user: FaceUser) => {
    setUsers((previous) => {
      const exists = previous.some((u) => u.id === user.id);
      const next = exists ? previous.map((u) => (u.id === user.id ? user : u)) : [...previous, user];
      return next.sort((a, b) => a.name.localeCompare(b.name));
    });
    // Counts in the status strip (enrolled, templates) change with every capture.
    getFaceStatus()
      .then(setStatus)
      .catch(() => undefined);
  }, []);

  const removeUser = useCallback((id: number) => {
    setUsers((previous) => previous.filter((u) => u.id !== id));
    getFaceStatus()
      .then(setStatus)
      .catch(() => undefined);
  }, []);

  return { status, users, error, loading, refresh, replaceUser, removeUser };
}
