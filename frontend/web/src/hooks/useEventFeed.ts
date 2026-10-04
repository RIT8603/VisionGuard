import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, toApiError } from '../api/client';
import { getEvents } from '../api/endpoints';
import type { FeedEvent } from '../api/types';
import { EVENTS_LIMIT, EVENTS_POLL_MS, MAX_FEED_EVENTS } from '../config';

interface EventFeedOptions {
  active: boolean;
  /** From usePipelineStatus; a change means event ids restarted at 1. */
  restartCount: number;
  /** Latest `event_count` from /api/status, used as a second restart check. */
  eventCount: number | null;
  /** Which camera's events to follow (null = the backend's first). Event ids are per camera. */
  camera?: string | null;
}

export interface EventFeed {
  /** Newest first, at most MAX_FEED_EVENTS. */
  events: FeedEvent[];
  error: ApiError | null;
}

/** Polls /api/events with a `since` cursor while `active`. */
export function useEventFeed({
  active,
  restartCount,
  eventCount,
  camera = null,
}: EventFeedOptions): EventFeed {
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [error, setError] = useState<ApiError | null>(null);

  const cursor = useRef(0);
  // Bumped on reset so a request started before the reset cannot merge
  // events from the previous backend run.
  const generation = useRef(0);
  const latestEventCount = useRef(eventCount);

  useEffect(() => {
    latestEventCount.current = eventCount;
  }, [eventCount]);

  // Event ids restart at 1 with the backend; a stale cursor would hide every
  // new event, so start over.
  const reset = useCallback(() => {
    generation.current += 1;
    cursor.current = 0;
    setEvents([]);
  }, []);

  useEffect(() => {
    if (restartCount > 0) reset();
  }, [restartCount, reset]);

  // Each camera numbers its own events, so another camera starts a new feed.
  const followed = useRef(camera);
  useEffect(() => {
    if (followed.current !== camera) {
      followed.current = camera;
      reset();
    }
  }, [camera, reset]);

  useEffect(() => {
    if (!active) return;

    const controller = new AbortController();
    let timer: number | undefined;

    const tick = async () => {
      const startedAt = performance.now();

      const count = latestEventCount.current;
      if (count !== null && count < cursor.current) reset();

      const requestGeneration = generation.current;
      try {
        const { events: batch } = await getEvents(
          cursor.current,
          EVENTS_LIMIT,
          controller.signal,
          camera,
        );
        if (requestGeneration === generation.current) {
          const fresh = batch.filter((event) => event.id > cursor.current);
          if (fresh.length > 0) {
            cursor.current = Math.max(cursor.current, ...fresh.map((event) => event.id));
            setEvents((previous) => {
              const seen = new Set(previous.map((event) => event.id));
              const added = fresh.filter((event) => !seen.has(event.id));
              return [...added, ...previous].slice(0, MAX_FEED_EVENTS);
            });
          }
          setError(null);
        }
      } catch (caught) {
        if (controller.signal.aborted) return;
        setError(toApiError(caught));
      }

      const elapsed = performance.now() - startedAt;
      timer = window.setTimeout(tick, Math.max(0, EVENTS_POLL_MS - elapsed));
    };

    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [active, camera, reset]);

  return { events, error };
}
