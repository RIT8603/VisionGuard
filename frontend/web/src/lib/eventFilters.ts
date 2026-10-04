import type { FeedEvent, Severity } from '../api/types';
import { severityOf } from './format';

export type EventSortOrder = 'newest' | 'oldest';

export interface EventFilters {
  severity: Severity | 'all';
  module: string;
  search: string;
  sort: EventSortOrder;
}

export const DEFAULT_EVENT_FILTERS: EventFilters = {
  severity: 'all',
  module: 'all',
  search: '',
  sort: 'newest',
};

export function filterEvents(events: FeedEvent[], filters: EventFilters): FeedEvent[] {
  const query = filters.search.trim().toLocaleLowerCase();

  return events
    .filter((event) => filters.severity === 'all' || severityOf(event.severity) === filters.severity)
    .filter((event) => filters.module === 'all' || event.module === filters.module)
    .filter((event) => {
      if (!query) return true;
      const searchable = [
        event.event,
        event.module,
        event.severity,
        event.timestamp,
        event.frame_id,
        event.id,
        JSON.stringify(event.subject ?? {}),
        JSON.stringify(event.data ?? {}),
      ]
        .join(' ')
        .toLocaleLowerCase();
      return searchable.includes(query);
    })
    .sort((left, right) => {
      const timeDifference = Date.parse(left.timestamp) - Date.parse(right.timestamp);
      const chronological =
        Number.isNaN(timeDifference) || timeDifference === 0
          ? left.id - right.id
          : timeDifference;
      return filters.sort === 'newest' ? -chronological : chronological;
    });
}
