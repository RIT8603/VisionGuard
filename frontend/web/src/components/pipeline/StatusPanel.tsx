import type { ReactNode } from 'react';
import type { FrameSummary } from '../../api/types';
import type { PipelineStatus } from '../../hooks/usePipelineStatus';
import { DASH, fmtDuration } from '../../lib/format';
import { Panel } from '../ui/Panel';
import { StateBadge } from './StateBadge';

interface StatusPanelProps {
  status: PipelineStatus;
  frame: FrameSummary | null;
}

export function StatusPanel({ status, frame }: StatusPanelProps) {
  const { data, viewState, lastSuccessAt, checkedAt } = status;
  const info = data?.info;

  const rows: [label: string, value: ReactNode][] = [
    ['State', <StateBadge state={viewState} />],
    ['Source', info?.source_type ?? DASH],
    ['Source id', info?.source_id ?? DASH],
    ['Frame size', frame ? `${frame.resolution.width} × ${frame.resolution.height}` : DASH],
    ['Processing', info?.resolution ?? DASH],
    ['Frame id', frame ? String(frame.frame_id) : DASH],
    ['Uptime', data ? fmtDuration(data.uptime_sec) : DASH],
    [
      'Last update',
      lastSuccessAt === null ? DASH : `${fmtDuration((checkedAt - lastSuccessAt) / 1000)} ago`,
    ],
  ];

  return (
    <Panel title="Source">
      <dl className="rows">
        {rows.map(([label, value]) => (
          <div className="row" key={label}>
            <dt>{label}</dt>
            <dd className="mono">{value}</dd>
          </div>
        ))}
      </dl>
    </Panel>
  );
}
