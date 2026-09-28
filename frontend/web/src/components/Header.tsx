import type { RunInfo } from '../api/types';
import { DASH, fmtDuration } from '../lib/format';
import type { ViewState } from '../state/viewState';
import { Wordmark } from './Logo';
import { StateBadge } from './StateBadge';

interface HeaderProps {
  info: RunInfo | undefined;
  viewState: ViewState;
  uptimeSec: number | undefined;
}

export function Header({ info, viewState, uptimeSec }: HeaderProps) {
  const readouts: [label: string, value: string][] = [
    ['Source', info?.source_id ?? DASH],
    ['Type', info?.source_type ?? DASH],
    ['Uptime', uptimeSec === undefined ? DASH : fmtDuration(uptimeSec)],
    ['Backend', info ? `v${info.version}` : DASH],
  ];

  return (
    <header className="topbar">
      <div className="brand">
        <h1>
          <Wordmark />
        </h1>
        <span className="brand-sub">Ops</span>
      </div>
      <StateBadge state={viewState} size="lg" />
      <dl className="readouts">
        {readouts.map(([label, value]) => (
          <div className="readout" key={label}>
            <dt>{label}</dt>
            <dd className="mono">{value}</dd>
          </div>
        ))}
      </dl>
    </header>
  );
}
