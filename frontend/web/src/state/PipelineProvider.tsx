import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import { hasFrame, hasStats } from '../api/guards';
import type { FrameSummary, PipelineStats } from '../api/types';
import { useCameras, type CameraList } from '../hooks/useCameras';
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
  /** Every camera and what it is doing; refreshed once a second. */
  cameras: CameraList;
  /** The camera the dashboard is following (id), or null before the list loads. */
  camera: string | null;
  /** `camera` as the API wants it: null for the backend's default (first) camera. */
  cameraParam: string | null;
  /** Follow another camera. Unknown ids fall back to the first camera. */
  selectCamera: (id: string) => void;
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
  const cameras = useCameras(visible);

  // The operator's choice, kept only while that camera still exists.
  const [selected, setSelected] = useState<string | null>(null);
  const selectCamera = useCallback((id: string) => setSelected(id), []);
  const list = cameras.data?.cameras ?? [];
  const firstId = list[0]?.id ?? null;
  const camera = selected !== null && list.some((c) => c.id === selected) ? selected : firstId;
  // The backend's default *is* its first camera, so ask for it by omission:
  // the status and event loops then do not restart when the list first loads.
  const cameraParam = camera === firstId ? null : camera;

  const status = usePipelineStatus(visible, cameraParam);
  const reachable = isReachable(status.viewState);

  const { data } = status;
  const frame = data && hasFrame(data.frame) ? data.frame : null;
  const stats = data && hasStats(data.stats) ? data.stats : null;

  const feed = useEventFeed({
    active: visible && reachable,
    restartCount: status.restartCount,
    eventCount: data?.event_count ?? null,
    camera: cameraParam,
  });

  // Not memoised: status is a new object on every poll, so a memo would
  // never hit. Consumers re-render at the poll rate either way.
  const value: PipelineContextValue = {
    status,
    feed,
    visible,
    reachable,
    frame,
    stats,
    cameras,
    camera,
    cameraParam,
    selectCamera,
  };

  return <PipelineContext value={value}>{children}</PipelineContext>;
}

export function usePipeline(): PipelineContextValue {
  const value = useContext(PipelineContext);
  if (value === null) {
    throw new Error('usePipeline() must be used inside <PipelineProvider>');
  }
  return value;
}
