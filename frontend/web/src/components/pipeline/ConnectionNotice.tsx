import type { ReactNode } from 'react';
import type { PipelineStatus } from '../../hooks/usePipelineStatus';
import { fmtDuration } from '../../lib/format';

function Notice({ tone, word, children }: { tone: 'warn' | 'crit'; word: string; children: ReactNode }) {
  return (
    <div className={`notice notice-${tone}`} role={tone === 'crit' ? 'alert' : 'status'}>
      <span className="notice-word">{word}</span>
      <div className="notice-body">{children}</div>
    </div>
  );
}

/** Explains the states where the user has to do something, or wait. */
export function ConnectionNotice({ status }: { status: PipelineStatus }) {
  const { viewState, error, lastSuccessAt, checkedAt, staleForMs } = status;

  switch (viewState) {
    case 'offline': {
      const since =
        lastSuccessAt !== null ? ` Last contact ${fmtDuration((checkedAt - lastSuccessAt) / 1000)} ago.` : '';
      return (
        <Notice tone="crit" word="Offline">
          <strong>Backend unreachable.</strong> {error ? `${error.message}.` : ''}
          {since} Start it from the repo root with{' '}
          <code>python -m backend.main --no-window --dashboard</code>. Retrying automatically.
        </Notice>
      );
    }
    case 'stopped':
      return (
        <Notice tone="warn" word="Stopped">
          <strong>Pipeline stopped.</strong> The backend exits when its pipeline ends (video finished,
          camera lost, or Ctrl+C). Restart it to continue.
        </Notice>
      );
    case 'stale':
      return (
        <Notice tone="warn" word="Stale">
          <strong>No new frames for {fmtDuration(staleForMs / 1000)}.</strong> The camera or stream may
          have stalled. The backend stops itself after 30 s without frames by default.
        </Notice>
      );
    default:
      return null;
  }
}
