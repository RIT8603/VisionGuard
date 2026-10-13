import type { ReactNode } from 'react';
import type { FrameSummary as Frame } from '../../api/types';
import { Panel } from '../ui/Panel';

interface FrameSummaryProps {
  frame: Frame | null;
  /** Rendered in the panel header, e.g. a link to the full tables. */
  action?: ReactNode;
}

interface Row {
  module: string;
  count: number;
  noun: string;
}

/** Counts per producing module: the overview version of FrameResults. */
export function FrameSummary({ frame, action }: FrameSummaryProps) {
  const rows: Row[] = frame
    ? [
        ...Object.entries(frame.tracks).map(([module, list]) => ({ module, count: list.length, noun: 'track' })),
        ...Object.entries(frame.detections).map(([module, list]) => ({ module, count: list.length, noun: 'detection' })),
        ...Object.entries(frame.poses).map(([module, list]) => ({ module, count: list.length, noun: 'pose' })),
      ]
    : [];

  return (
    <Panel
      title="Current frame"
      aside={
        <>
          {frame && <span className="mono">F{frame.frame_id}</span>}
          {action}
        </>
      }
    >
      {!frame ? (
        <p className="empty">No frame yet.</p>
      ) : rows.length === 0 ? (
        <p className="empty">No detection or tracking module produced results.</p>
      ) : (
        <ul className="frame-summary">
          {rows.map((row) => (
            <li key={`${row.noun}:${row.module}`}>
              <span className="mono frame-summary-module">{row.module}</span>
              <span className="mono frame-summary-count">
                {row.count}
                <span className="unit">
                  {row.noun}
                  {row.count === 1 ? '' : 's'}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
