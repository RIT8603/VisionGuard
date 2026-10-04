import { useState } from 'react';
import type { Severity } from '../../api/types';
import { EventFilters as EventFiltersControl } from '../../components/events/EventFilters';
import { EventFeed } from '../../components/pipeline/EventFeed';
import { Panel } from '../../components/ui/Panel';
import { usePipeline } from '../../state/PipelineProvider';
import { DEFAULT_EVENT_FILTERS, filterEvents, type EventFilters } from '../../lib/eventFilters';
import { severityOf } from '../../lib/format';

const SEVERITIES: Severity[] = ['info', 'warning', 'critical'];

export function EventsPage() {
  const { feed, reachable } = usePipeline();
  const [filters, setFilters] = useState<EventFilters>(DEFAULT_EVENT_FILTERS);
  const filteredEvents = filterEvents(feed.events, filters);
  const modulesReporting = new Set(feed.events.map((event) => event.module)).size;
  const counts = feed.events.reduce<Record<Severity, number>>(
    (total, event) => {
      total[severityOf(event.severity)] += 1;
      return total;
    },
    { info: 0, warning: 0, critical: 0 },
  );
  const modules = [...new Set(feed.events.map((event) => event.module))].sort((a, b) =>
    a.localeCompare(b),
  );

  return (
    <>
      <title>Events &amp; Alerts · VisionGuard</title>
      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Events &amp; Alerts</h2>
          <p className="page-intro">Review the events currently available from the pipeline.</p>
        </header>

        <div className="notice notice-info event-notice" role="note">
          <span className="notice-word">Session-only</span>
          <div className="notice-body">
            Events are held in the backend’s in-memory buffer and are not persisted beyond that
            buffer. Notification delivery is not implemented.
          </div>
        </div>

        <Panel title="Event summary" className="event-summary-panel">
          <div className="event-summary">
            <SummaryValue label="Total" value={feed.events.length} />
            {SEVERITIES.map((severity) => (
              <SummaryValue key={severity} label={severity} value={counts[severity]} />
            ))}
            <SummaryValue label="Modules reporting" value={modulesReporting} />
          </div>
        </Panel>

        <EventFiltersControl filters={filters} modules={modules} onChange={setFilters} />

        <EventFeed
          events={filteredEvents}
          error={reachable ? feed.error : null}
          title="Event feed"
          emptyMessage={
            feed.events.length === 0 ? 'No events yet.' : 'No events match these filters'
          }
          showPayload
        />
      </div>
    </>
  );
}

function SummaryValue({ label, value }: { label: string; value: number }) {
  return (
    <div className="event-summary-value">
      <span className="tile-label">{label}</span>
      <span className="mono">{value}</span>
    </div>
  );
}
