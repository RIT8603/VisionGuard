import { useEffect, useRef, useState } from 'react';
import { ApiError, toApiError } from '../api/client';
import { getStatus } from '../api/endpoints';
import { hasFrame } from '../api/guards';
import type { StatusResponse } from '../api/types';
import { RESTART_TOLERANCE_SEC, REQUEST_TIMEOUT_MS, STATUS_POLL_MS } from '../config';
import { deriveViewState, type ViewState } from '../state/viewState';

export interface PipelineStatus {
  /** Last successful response, kept while offline so the UI can show it dimmed. */
  data: StatusResponse | null;
  /** Error from the most recent poll; null after a success. */
  error: ApiError | null;
  /** Consecutive failed polls. */
  failures: number;
  viewState: ViewState;
  /** Date.now() of the last successful poll. */
  lastSuccessAt: number | null;
  /** Milliseconds since frame_id last changed. */
  staleForMs: number;
  /** Increments whenever a backend restart is detected. */
  restartCount: number;
  /** Date.now() of the most recent poll, successful or not. */
  checkedAt: number;
}

const INITIAL: PipelineStatus = {
  data: null,
  error: null,
  failures: 0,
  viewState: 'loading',
  lastSuccessAt: null,
  staleForMs: 0,
  restartCount: 0,
  checkedAt: 0,
};

/**
 * Polls /api/status every STATUS_POLL_MS while `active`.
 *
 * `camera` picks which camera's status to follow (null = the backend's
 * first). Changing it starts over from "connecting" for the new camera.
 *
 * The next request is scheduled only after the previous one settles, so a
 * slow backend is never hit with overlapping requests.
 */
export function usePipelineStatus(active: boolean, camera: string | null = null): PipelineStatus {
  const [status, setStatus] = useState<PipelineStatus>(INITIAL);

  // Bookkeeping that survives pauses (tab hidden) without triggering renders.
  const tracking = useRef({
    data: null as StatusResponse | null,
    error: null as ApiError | null,
    failures: 0,
    lastSuccessAt: null as number | null,
    lastFrameId: null as number | null,
    lastFrameChangeAt: null as number | null,
    lastUptime: null as number | null,
    restartCount: 0,
    camera: undefined as string | null | undefined,
  });

  const isCameraSwitching = useRef(false);

  useEffect(() => {
    if (!active) return;

    const t = tracking.current;
    const controller = new AbortController();
    let timer: number | undefined;

    // A different camera is a different pipeline: its frame ids and uptime
    // have nothing to do with the last one's, so forget them (but not the
    // restart count - this is not a backend restart).
    const cameraChanged = t.camera !== camera;
    if (cameraChanged) {
      isCameraSwitching.current = true;
      t.camera = camera;
      t.data = null;
      t.error = null;
      t.failures = 0;
      t.lastSuccessAt = null;
      t.lastFrameId = null;
      t.lastFrameChangeAt = null;
      t.lastUptime = null;
    }

    const publish = () => {
      const now = Date.now();
      const staleForMs = t.lastFrameChangeAt === null ? 0 : now - t.lastFrameChangeAt;
      setStatus({
        data: t.data,
        error: t.error,
        failures: t.failures,
        lastSuccessAt: t.lastSuccessAt,
        staleForMs,
        restartCount: t.restartCount,
        checkedAt: now,
        viewState: deriveViewState({ data: t.data, failures: t.failures, staleForMs }),
      });
    };

    const tick = async () => {
      if (controller.signal.aborted) return;
      const startedAt = performance.now();
      const timeoutMs = isCameraSwitching.current ? 5000 : REQUEST_TIMEOUT_MS;

      try {
        const data = await getStatus(controller.signal, camera, timeoutMs);
        if (controller.signal.aborted || t.camera !== camera) return;
        const now = Date.now();

        // uptime_sec is wall-clock time since the dashboard started, so it
        // only goes backwards when a new backend process has replaced the old.
        if (t.lastUptime !== null && data.uptime_sec < t.lastUptime - RESTART_TOLERANCE_SEC) {
          t.restartCount += 1;
          t.lastFrameId = null;
        }
        t.lastUptime = data.uptime_sec;

        const frameId = hasFrame(data.frame) ? data.frame.frame_id : null;
        if (frameId !== t.lastFrameId) {
          t.lastFrameId = frameId;
          t.lastFrameChangeAt = now;
        }

        // Successfully received response: clear any offline and switching flags
        isCameraSwitching.current = false;
        t.data = data;
        t.error = null;
        t.failures = 0;
        t.lastSuccessAt = now;
      } catch (error) {
        if (controller.signal.aborted || t.camera !== camera) return;
        t.failures += 1;
        t.error = toApiError(error);
      }

      publish();
      const elapsed = performance.now() - startedAt;
      timer = window.setTimeout(tick, Math.max(0, STATUS_POLL_MS - elapsed));
    };

    if (cameraChanged) publish();
    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [active, camera]);

  return status;
}
