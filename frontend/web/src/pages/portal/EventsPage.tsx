import { useState } from 'react';
import { EventFilters } from '../../components/events/EventFilters';
import { EventFeed } from '../../components/pipeline/EventFeed';
import { Panel } from '../../components/ui/Panel';
import {
  countBySeverity,
  DEFAULT_FILTER,
  filterEvents,
  isDefaultFilter,
  modulesIn,
  type EventFilter,
} from '../../lib/eventFilters';
import { usePipeline } from '../../state/PipelineProvider';

/**
 * The event log at full length, with client-side filters. Reads the feed the
 * PipelineProvider already polls; filtering never triggers a request.
 */
export function EventsPage() {
  const { feed, reachable } = usePipeline();
  const [filter, setFilter] = useState<EventFilter>(DEFAULT_FILTER);

  const counts = countBySeverity(feed.events);
  const modules = modulesIn(feed.events);
  const shown = filterEvents(feed.events, filter);

  const tiles = [
    { label: 'In memory', value: feed.events.length, tone: undefined },
    { label: 'Info', value: counts.info, tone: 'primary' },
    { label: 'Warning', value: counts.warning, tone: counts.warning > 0 ? 'warn' : undefined },
    { label: 'Critical', value: counts.critical, tone: counts.critical > 0 ? 'crit' : undefined },
    { label: 'Modules reporting', value: modules.length, tone: undefined },
  ] as const;

  return (
    <>
      <title>Events &amp; Alerts · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Events &amp; alerts</h2>
          <p className="page-intro">
            What the analysis modules concluded, such as zone entries, dwell time or possible falls,
            as the backend reports them.
          </p>
        </header>

        <div className="notice notice-info" role="note">
          <span className="notice-word">Session</span>
          <div className="notice-body">
            Shows up to the 200 most recent events: what the backend still holds when the portal
            opens, then new ones as they arrive. Nothing is saved, and no notifications are sent.
          </div>
        </div>

        <div className="events-layout">
          <Panel title="Summary" className="panel-flush">
            <div className="tiles">
              {tiles.map((tile) => (
                <div className={tile.tone ? `tile tile-${tile.tone}` : 'tile'} key={tile.label}>
                  <div className="tile-label">{tile.label}</div>
                  <div className="tile-value mono">{tile.value}</div>
                </div>
              ))}
            </div>
          </Panel>

          <EventFilters value={filter} onChange={setFilter} modules={modules} counts={counts} />

          <EventFeed
            events={shown}
            error={reachable ? feed.error : null}
            title={isDefaultFilter(filter) ? 'Event log' : 'Event log (filtered)'}
            className="panel-log-page"
            emptyMessage={
              feed.events.length === 0
                ? undefined
                : 'No events match these filters.'
            }
          />
        </div>
      </div>
    </>
  );
}
