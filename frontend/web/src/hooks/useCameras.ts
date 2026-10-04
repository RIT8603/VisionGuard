import { useEffect, useState } from 'react';
import { ApiError, toApiError } from '../api/client';
import { getCameras } from '../api/endpoints';
import type { CamerasResponse } from '../api/types';
import { CAMERAS_POLL_MS } from '../config';

export interface CameraList {
  /** Last successful response; kept while the backend is away. */
  data: CamerasResponse | null;
  error: ApiError | null;
}

/**
 * Polls /api/cameras every CAMERAS_POLL_MS while `active`.
 *
 * Like the status loop, the next request is scheduled only after the
 * previous one settles.
 */
export function useCameras(active: boolean): CameraList {
  const [state, setState] = useState<CameraList>({ data: null, error: null });

  useEffect(() => {
    if (!active) return;

    const controller = new AbortController();
    let timer: number | undefined;

    const tick = async () => {
      const startedAt = performance.now();
      try {
        const data = await getCameras(controller.signal);
        setState({ data, error: null });
      } catch (caught) {
        if (controller.signal.aborted) return;
        setState((previous) => ({ data: previous.data, error: toApiError(caught) }));
      }
      const elapsed = performance.now() - startedAt;
      timer = window.setTimeout(tick, Math.max(0, CAMERAS_POLL_MS - elapsed));
    };

    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [active]);

  return state;
}
