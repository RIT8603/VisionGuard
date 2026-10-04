import type { CameraInfo, RunInfo } from '../../api/types';
import { DASH, fmtDuration } from '../../lib/format';
import type { ViewState } from '../../state/viewState';
import { Wordmark } from '../brand/Logo';
import { StateBadge } from '../pipeline/StateBadge';

interface PortalHeaderProps {
  info: RunInfo | undefined;
  viewState: ViewState;
  uptimeSec: number | undefined;
  /** All cameras; the header adds a "Cameras" readout when there are several. */
  cameras?: readonly CameraInfo[];
}

export function PortalHeader({ info, viewState, uptimeSec, cameras = [] }: PortalHeaderProps) {
  const live = cameras.filter((camera) => camera.status === 'live').length;
  const readouts: [label: string, value: string][] = [
    ...(cameras.length > 1
      ? ([['Cameras', `${live} of ${cameras.length} live`]] as [string, string][])
      : []),
    ['Source', info?.camera_name ?? info?.source_id ?? DASH],
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
