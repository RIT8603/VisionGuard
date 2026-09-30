import { createContext, useContext, type ReactNode } from 'react';
import { hasFrame, hasStats } from '../api/guards';
import type { FrameSummary, PipelineStats } from '../api/types';
import { useEventFeed, type EventFeed } from '../hooks/useEventFeed';
import { usePageVisible } from '../hooks/usePageVisible';
import { usePipelineStatus, type PipelineStatus } from '../hooks/usePipelineStatus';
import { isReachable } from './viewState';

export interface PipelineContextValue {
  status: PipelineStatus;
  feed: EventFeed;
  /** False while the tab is hidden; polling and the stream pause. */
  visible: boolean;
  /** The backend is answering (waiting, live or stale). */
  reachable: boolean;
  /** Current frame, or null before the first one. */
  frame: FrameSummary | null;
  /** Pipeline stats, or null before the first frame. */
  stats: PipelineStats | null;
}

const PipelineContext = createContext<PipelineContextValue | null>(null);

/**
 * Runs the status and event polling once for everything beneath it.
 *
 * Mount it around the portal only: public pages then never poll, and every
 * portal page shares one status loop, one event cursor and one restart
 * detector. Unmounting it (leaving the portal) stops both loops.
 */
export function PipelineProvider({ children }: { children: ReactNode }) {
  const visible = usePageVisible();
  const status = usePipelineStatus(visible);
  const reachable = isReachable(status.viewState);

  const { data } = status;
  const frame = data && hasFrame(data.frame) ? data.frame : null;
  const stats = data && hasStats(data.stats) ? data.stats : null;

  const feed = useEventFeed({
    active: visible && reachable,
    restartCount: status.restartCount,
    eventCount: data?.event_count ?? null,
  });

  // Not memoised: status is a new object on every poll, so a memo would
  // never hit. Consumers re-render at the poll rate either way.
  const value: PipelineContextValue = { status, feed, visible, reachable, frame, stats };

  return <PipelineContext value={value}>{children}</PipelineContext>;
}

export function usePipeline(): PipelineContextValue {
  const value = useContext(PipelineContext);
  if (value === null) {
    throw new Error('usePipeline() must be used inside <PipelineProvider>');
  }
  return value;
}
