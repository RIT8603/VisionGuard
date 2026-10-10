import { useState } from 'react';
import { toApiError } from '../../api/client';
import { acknowledgeAlert, acknowledgeAllAlerts } from '../../api/endpoints';
import type { AlertInfo, AlertStatus, Severity } from '../../api/types';
import { useAlerts } from '../../hooks/useAlerts';
import { usePageVisible } from '../../hooks/usePageVisible';
import { describeData, describeSubject, fmtTime, severityOf } from '../../lib/format';

type Filter = AlertStatus | 'all';

const FILTERS: { value: Filter; label: string }[] = [
  { value: 'open', label: 'Open' },
  { value: 'acknowledged', label: 'Acknowledged' },
  { value: 'all', label: 'All' },
];

const SEVERITY_LABEL: Record<Severity, string> = {
  info: 'Info',
  warning: 'Warn',
  critical: 'Crit',
};

export function AlertsPage() {
  const visible = usePageVisible();
  const [filter, setFilter] = useState<Filter>('open');
  const { data, error, refresh } = useAlerts(visible, filter);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setActionError(null);
    try {
      await action();
    } catch (caught) {
      setActionError(toApiError(caught).message);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const alerts = data?.alerts ?? [];
  const openCount = data?.open_count ?? 0;

  return (
    <>
      <title>Alerts · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          {/* h2: the portal header's brand is the page's h1. */}
          <h2 className="page-title">Alerts</h2>
          <p className="page-intro">
            Events at or above <strong>{data?.min_severity ?? 'warning'}</strong> severity, from
            every camera. Repeats of the same event about the same subject fold into one alert,
            which stays open until someone acknowledges it.
          </p>
        </header>

        <div className="alerts-toolbar">
          <div className="alerts-filters" role="group" aria-label="Show">
            {FILTERS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`alerts-filter${filter === option.value ? ' is-active' : ''}`}
                aria-pressed={filter === option.value}
                onClick={() => setFilter(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <span className="page-meta mono">
            {openCount} open
            {data && data.open_critical > 0 && (
              <span className="text-crit"> · {data.open_critical} critical</span>
            )}
          </span>
          <button
            type="button"
            className="summary-btn"
            disabled={busy || openCount === 0}
            onClick={() => void run(() => acknowledgeAllAlerts())}
          >
            Acknowledge all
          </button>
        </div>

        {error && <p className="inline-error">Alerts unavailable: {error.message}</p>}
        {actionError && <p className="inline-error">{actionError}</p>}

        {alerts.length === 0 ? (
          <p className="empty">
            {filter === 'open'
              ? 'No open alerts. Alerts come from analysis modules such as restricted_zone, fall_detection or crowd_alert.'
              : 'No alerts to show.'}
          </p>
        ) : (
          <ol className="log alerts-list">
            {alerts.map((alert) => (
              <AlertRow
                key={alert.id}
                alert={alert}
                busy={busy}
                onAcknowledge={() => void run(() => acknowledgeAlert(alert.id))}
              />
            ))}
          </ol>
        )}
      </div>
    </>
  );
}

interface AlertRowProps {
  alert: AlertInfo;
  busy: boolean;
  onAcknowledge: () => void;
}

function AlertRow({ alert, busy, onAcknowledge }: AlertRowProps) {
  const severity = severityOf(alert.severity);
  const subject = describeSubject(alert.subject);
  const detail = describeData(alert.data);
  const open = alert.status === 'open';

  return (
    <li className={`log-row sev-${severity}${open ? '' : ' is-acknowledged'}`}>
      <span className="log-sev">{SEVERITY_LABEL[severity]}</span>
      <div className="log-main">
        <div className="log-line">
          <span className="log-event">{alert.event}</span>
          {subject && <span className="log-subject mono">{subject}</span>}
          {alert.count > 1 && <span className="alert-count mono">×{alert.count}</span>}
        </div>
        {detail && <div className="log-detail">{detail}</div>}
        <div className="log-meta mono">
          <span>{alert.camera}</span>
          <time dateTime={alert.last_seen}>{fmtTime(alert.last_seen)}</time>
          {alert.count > 1 && <span>since {fmtTime(alert.first_seen)}</span>}
          <span>{alert.module}</span>
          {!open && alert.acknowledged_at && <span>ack {fmtTime(alert.acknowledged_at)}</span>}
        </div>
      </div>
      {open && (
        <button type="button" className="summary-btn summary-btn-quiet alert-ack" disabled={busy} onClick={onAcknowledge}>
          Acknowledge
        </button>
      )}
    </li>
  );
}
