import type { PipelineStats } from '../api/types';
import { DASH, fmtNumber } from '../lib/format';
import { Panel } from './Panel';

interface StatsPanelProps {
  stats: PipelineStats | null;
  eventCount: number | null;
}

interface Tile {
  label: string;
  value: string;
  unit?: string;
  tone?: 'primary' | 'warn';
}

export function StatsPanel({ stats, eventCount }: StatsPanelProps) {
  const failed = stats?.frames_failed ?? 0;

  const tiles: Tile[] = [
    { label: 'FPS', value: fmtNumber(stats?.fps, 1), tone: 'primary' },
    { label: 'Avg FPS', value: fmtNumber(stats?.overall_fps, 1) },
    { label: 'Latency', value: fmtNumber(stats?.latency_ms, 0), unit: 'ms', tone: 'primary' },
    { label: 'Capture wait', value: fmtNumber(stats?.capture_ms, 1), unit: 'ms' },
    { label: 'Frames', value: fmtNumber(stats?.frames_processed) },
    { label: 'Failed frames', value: fmtNumber(stats?.frames_failed), tone: failed > 0 ? 'warn' : undefined },
    { label: 'Events', value: eventCount === null ? DASH : String(eventCount) },
  ];

  return (
    <Panel title="Telemetry" className="panel-flush">
      <div className="tiles">
        {tiles.map(({ label, value, unit, tone }) => (
          <div className={tone ? `tile tile-${tone}` : 'tile'} key={label}>
            <div className="tile-label">{label}</div>
            <div className="tile-value mono">
              {value}
              {unit && value !== DASH && <span className="tile-unit">{unit}</span>}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
