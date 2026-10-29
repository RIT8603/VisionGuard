import { Link } from 'react-router';
import type { WakeStage, WakeStatus } from '../../api/wake';
import { PATHS } from '../../app/paths';
import { LogoMark } from '../brand/Logo';

function fmtSeconds(total: number): string {
  const s = Math.max(0, Math.round(total));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

function StageRow({ stage }: { stage: WakeStage }) {
  const mark = { done: '✓', active: '', pending: '', failed: '✕' }[stage.status];
  const timing =
    stage.seconds !== undefined
      ? fmtSeconds(stage.seconds)
      : stage.status === 'pending'
        ? `~${fmtSeconds(stage.typical_seconds)}`
        : '';
  return (
    <li className={`wake-stage wake-stage-${stage.status}`}>
      <span className="wake-stage-mark" aria-hidden="true">
        {mark}
      </span>
      <span className="wake-stage-label">
        {stage.label}
        <span className="visually-hidden"> ({stage.status})</span>
      </span>
      <span className="wake-stage-time">{timing}</span>
    </li>
  );
}

interface StartupScreenProps {
  /** Latest report from the wake service; null until the first one arrives. */
  status: WakeStatus | null;
  /** Ask again after an error (capacity, timeout, refused start). */
  onRetry: () => void;
}

/**
 * Shown instead of the portal while the GPU backend starts. Every number on it
 * comes from the wake service's real checks: the bar fills only when a stage
 * is verified, and the moving stripe marks "working", not progress.
 */
export function StartupScreen({ status, onRetry }: StartupScreenProps) {
  const progress = status ? Math.min(1, Math.max(0, status.progress)) : 0;
  const pct = Math.round(progress * 100);
  const working = status !== null && !status.error && !status.ready;
  const eta = status?.eta_seconds;

  let etaText = 'Checking the GPU backend…';
  if (status?.error) etaText = 'Startup stopped';
  else if (status?.state === 'stopping') etaText = 'Waiting for the previous shutdown to finish';
  else if (eta !== null && eta !== undefined) etaText = eta > 0 ? `About ${fmtSeconds(eta)} left` : 'Taking longer than usual…';
  else if (status) etaText = 'Waiting for the GPU server to start';

  return (
    <div className="wake-screen">
      <section className="wake-phone" aria-labelledby="wake-title" aria-busy={working}>
        <div className="wake-notch" aria-hidden="true" />
        <LogoMark size={44} className="wake-logo" />
        <h1 id="wake-title" className="wake-title">
          Starting the AI backend
        </h1>
        <p className="wake-lede">
          The GPU server sleeps when nobody is using it, to save cost. It wakes up now and the
          dashboard opens by itself once the model is running.
        </p>

        <div
          className="wake-bar"
          role="progressbar"
          aria-label="Startup stages verified"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
        >
          <div className="wake-bar-fill" style={{ width: `${pct}%` }} />
          {working && <div className="wake-bar-stripe" aria-hidden="true" />}
        </div>
        <div className="wake-bar-meta">
          <span>{pct}% of stages done</span>
          <span>{status?.elapsed_seconds != null ? `${fmtSeconds(status.elapsed_seconds)} elapsed` : ''}</span>
        </div>

        <p className="wake-status" role="status" aria-live="polite">
          {working && <span className="wake-spinner" aria-hidden="true" />}
          <strong>{status?.message ?? 'Contacting the wake service'}</strong>
        </p>
        <p className="wake-eta">
          {etaText}
          {status && !status.error && eta != null && <span className="wake-basis"> · {status.eta_basis}</span>}
        </p>

        {status?.error && (
          <div className="notice notice-warn wake-error" role="alert">
            <span className="notice-word">Problem</span>
            <div className="notice-body">
              {status.error}
              <button type="button" className="summary-btn wake-retry" onClick={onRetry}>
                Try again
              </button>
            </div>
          </div>
        )}

        {status && (
          <ol className="wake-stages" aria-label="Startup stages">
            {status.stages.map((stage) => (
              <StageRow key={stage.id} stage={stage} />
            ))}
          </ol>
        )}

        <p className="wake-foot">
          While you wait: <Link to={PATHS.home}>Home</Link> · <Link to={PATHS.about}>About</Link> ·{' '}
          <Link to={PATHS.contact}>Contact</Link>
          {status && <> · It sleeps again after {status.idle_minutes} idle minutes.</>}
        </p>
      </section>
    </div>
  );
}
