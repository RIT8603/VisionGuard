import type { FrameSummary as FrameSummaryData } from '../../api/types';
import { Panel } from '../ui/Panel';

interface FrameSummaryProps {
  frame: FrameSummaryData | null;
}

export function FrameSummary({ frame }: FrameSummaryProps) {
  if (!frame) {
    return (
      <Panel title="Current frame">
        <p className="empty">No frame yet.</p>
      </Panel>
    );
  }

  const modules = new Set([
    ...Object.keys(frame.detections),
    ...Object.keys(frame.tracks),
    ...Object.keys(frame.poses),
    ...Object.keys(frame.errors),
  ]);

  return (
    <Panel
      title="Current frame"
      aside={
        <span className="mono">
          F{frame.frame_id} · {frame.resolution.width}×{frame.resolution.height}
        </span>
      }
    >
      {modules.size === 0 ? (
        <p className="empty">No module results for this frame.</p>
      ) : (
        <ul className="frame-summary">
          {[...modules].map((module) => {
            const detections = frame.detections[module]?.length ?? 0;
            const tracks = frame.tracks[module]?.length ?? 0;
            const poses = frame.poses[module]?.length ?? 0;
            const error = frame.errors[module];

            return (
              <li className="frame-summary-row" key={module}>
                <span className="mono frame-summary-module">{module}</span>
                <span className="frame-summary-counts">
                  {detections} detection{detections === 1 ? '' : 's'} · {tracks} track
                  {tracks === 1 ? '' : 's'} · {poses} pose{poses === 1 ? '' : 's'}
                </span>
                {error && <span className="module-error mono">{error}</span>}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
