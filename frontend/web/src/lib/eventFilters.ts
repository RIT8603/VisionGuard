import type { FeedEvent, Severity } from '../api/types';
import { describeData, describeSubject, severityOf } from './format';

/** Client-side filtering over the events already held in memory. */

export const SEVERITIES: readonly Severity[] = ['info', 'warning', 'critical'];

export type EventOrder = 'newest' | 'oldest';

export interface EventFilter {
  severities: ReadonlySet<Severity>;
  /** null = every module. */
  module: string | null;
  query: string;
  order: EventOrder;
}

export const DEFAULT_FILTER: EventFilter = {
  severities: new Set(SEVERITIES),
  module: null,
  query: '',
  order: 'newest',
};

export function isDefaultFilter(filter: EventFilter): boolean {
  return (
    filter.severities.size === SEVERITIES.length &&
    filter.module === null &&
    filter.query.trim() === '' &&
    filter.order === 'newest'
  );
}

/** Everything a person might type to find an event. */
function searchText(event: FeedEvent): string {
  return [event.event, event.module, describeSubject(event.subject), describeData(event.data)]
    .join(' ')
    .toLowerCase();
}

/** `events` arrive newest first, as useEventFeed provides them. */
export function filterEvents(events: readonly FeedEvent[], filter: EventFilter): FeedEvent[] {
  const query = filter.query.trim().toLowerCase();
  const matches = events.filter(
    (event) =>
      filter.severities.has(severityOf(event.severity)) &&
      (filter.module === null || event.module === filter.module) &&
      (query === '' || searchText(event).includes(query)),
  );
  return filter.order === 'oldest' ? matches.reverse() : matches;
}

export function countBySeverity(events: readonly FeedEvent[]): Record<Severity, number> {
  const counts: Record<Severity, number> = { info: 0, warning: 0, critical: 0 };
  for (const event of events) counts[severityOf(event.severity)] += 1;
  return counts;
}

export function modulesIn(events: readonly FeedEvent[]): string[] {
  return [...new Set(events.map((event) => event.module))].sort();
}
