import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, toApiError } from '../api/client';
import { getAlerts } from '../api/endpoints';
import type { AlertsResponse, AlertStatus } from '../api/types';
import { ALERTS_POLL_MS } from '../config';

export interface AlertList {
  /** Last successful response; kept while the backend is away. */
  data: AlertsResponse | null;
  error: ApiError | null;
  /** Fetch now instead of waiting for the next poll, e.g. after acknowledging. */
  refresh: () => void;
}

/**
 * Polls /api/alerts every ALERTS_POLL_MS while `active`. Like the other
 * loops, the next request is scheduled only after the previous one settles.
 */
export function useAlerts(active: boolean, status: AlertStatus | 'all'): AlertList {
  const [state, setState] = useState<Omit<AlertList, 'refresh'>>({ data: null, error: null });
  const kick = useRef<() => void>(() => undefined);

  useEffect(() => {
    if (!active) return;

    const controller = new AbortController();
    let timer: number | undefined;

    const tick = async () => {
      window.clearTimeout(timer);
      const startedAt = performance.now();
      try {
        const data = await getAlerts(status, controller.signal);
        setState({ data, error: null });
      } catch (caught) {
        if (controller.signal.aborted) return;
        setState((previous) => ({ data: previous.data, error: toApiError(caught) }));
      }
      const elapsed = performance.now() - startedAt;
      timer = window.setTimeout(tick, Math.max(0, ALERTS_POLL_MS - elapsed));
    };

    kick.current = () => void tick();
    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
      kick.current = () => undefined;
    };
  }, [active, status]);

  const refresh = useCallback(() => kick.current(), []);
  return { ...state, refresh };
}
