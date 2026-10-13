import type { ReactNode } from 'react';
import type { ApiError } from '../../api/client';
import type { FeedEvent, Severity } from '../../api/types';
import { describeData, describeSubject, fmtTime, severityOf } from '../../lib/format';
import { Panel } from '../ui/Panel';

const DEFAULT_EMPTY =
  'No events yet. Events come from analysis modules such as restricted_zone or fall_detection.';

interface EventFeedProps {
  events: FeedEvent[];
  error: ApiError | null;
  title?: string;
  /** Extra panel class, e.g. to let the log grow on a dedicated page. */
  className?: string;
  /** Shown when `events` is empty, e.g. "nothing matches these filters". */
  emptyMessage?: string;
  /** Rendered in the panel header after the count, e.g. a link. */
  action?: ReactNode;
  /** Show only the newest N; the header still counts all of them. */
  limit?: number;
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
  className,
  emptyMessage = DEFAULT_EMPTY,
  action,
  limit,
}: EventFeedProps) {
  const critical = events.filter((e) => severityOf(e.severity) === 'critical').length;
  const shown = limit === undefined ? events : events.slice(0, limit);

  return (
    <Panel
      title={title}
      className={className ? `panel-log ${className}` : 'panel-log'}
      aside={
        <span className="mono">
          {critical > 0 && <span className="text-crit">{critical} crit · </span>}
          {events.length}
          {action}
        </span>
      }
    >
      {error && <p className="inline-error">Events unavailable: {error.message}</p>}
      {events.length === 0 ? (
        <p className="empty">{emptyMessage}</p>
      ) : (
        <ol className="log">
          {shown.map((event) => {
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
                    <time dateTime={event.timestamp}>{fmtTime(event.timestamp)}</time>
                    <span>F{event.frame_id}</span>
                    <span>{event.module}</span>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
