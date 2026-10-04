import type { ApiError } from '../../api/client';
import type { FeedEvent, Severity } from '../../api/types';
import { describeData, describeSubject, fmtTime, severityOf } from '../../lib/format';
import { Panel } from '../ui/Panel';

interface EventFeedProps {
  events: FeedEvent[];
  error: ApiError | null;
  title?: string;
  emptyMessage?: string;
  showPayload?: boolean;
}

const SEVERITY_LABEL: Record<Severity, string> = {
  info: 'Info',
  warning: 'Warn',
  critical: 'Crit',
};

export function EventFeed({
  events,
  error,
  title = 'Event log',
  emptyMessage,
  showPayload = false,
}: EventFeedProps) {
  const critical = events.filter((e) => severityOf(e.severity) === 'critical').length;

  return (
    <Panel
      title={title}
      className="panel-log"
      aside={
        <span className="mono">
          {critical > 0 && <span className="text-crit">{critical} crit · </span>}
          {events.length}
        </span>
      }
    >
      {error && <p className="inline-error">Events unavailable: {error.message}</p>}
      {events.length === 0 ? (
        <p className="empty">
          {emptyMessage ??
            'No events yet. Events come from analysis modules such as restricted_zone or fall_detection.'}
        </p>
      ) : (
        <ol className="log">
          {events.map((event) => {
            const severity = severityOf(event.severity);
            const subject = describeSubject(event.subject);
            const detail = describeData(event.data);
            return (
              <li key={event.id} className={`log-row sev-${severity}`}>
                <span className="log-sev">{SEVERITY_LABEL[severity]}</span>
                <div className="log-main">
                  <div className="log-line">
                    <span className="log-event">{event.event}</span>
                    {subject && <span className="log-subject mono">{subject}</span>}
                  </div>
                  {detail && <div className="log-detail">{detail}</div>}
                  <div className="log-meta mono">
                    <time dateTime={event.timestamp}>
                      {formatTimestamp(event.timestamp, showPayload)}
                    </time>
                    <span>F{event.frame_id}</span>
                    <span>{event.module}</span>
                  </div>
                  {showPayload && (
                    <details className="event-payload">
                      <summary>Full event payload · #{event.id}</summary>
                      <pre>{JSON.stringify(event, null, 2)}</pre>
                    </details>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}

function formatTimestamp(timestamp: string, includeDate: boolean): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return fmtTime(timestamp);
  return includeDate ? date.toLocaleString() : fmtTime(timestamp);
}
