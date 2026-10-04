import type { Severity } from '../../api/types';
import type { EventFilters as EventFiltersState, EventSortOrder } from '../../lib/eventFilters';

interface EventFiltersProps {
  filters: EventFiltersState;
  modules: string[];
  onChange: (filters: EventFiltersState) => void;
}

export function EventFilters({ filters, modules, onChange }: EventFiltersProps) {
  const update = <K extends keyof EventFiltersState>(key: K, value: EventFiltersState[K]) =>
    onChange({ ...filters, [key]: value });
  const updateSeverity = (value: string) => {
    const severity: Severity | 'all' =
      value === 'warning' || value === 'critical' || value === 'info' ? value : 'all';
    update('severity', severity);
  };
  const updateSort = (value: string) => {
    const sort: EventSortOrder = value === 'oldest' ? 'oldest' : 'newest';
    update('sort', sort);
  };

  return (
    <section className="event-filters" aria-label="Filter events">
      {/* Placeholder until events carry a camera id: no values are offered or invented. */}
      <label className="filter-field filter-camera">
        <span>Camera</span>
        <select value="all" disabled aria-describedby="camera-filter-hint">
          <option value="all">All (single source)</option>
        </select>
        <span id="camera-filter-hint" className="filter-hint">
          Needs backend support: events do not carry a camera id yet.
        </span>
      </label>
      <label className="filter-field">
        <span>Severity</span>
        <select
          value={filters.severity}
          onChange={(event) => updateSeverity(event.target.value)}
        >
          <option value="all">All severities</option>
          <option value="info">Info</option>
          <option value="warning">Warning</option>
          <option value="critical">Critical</option>
        </select>
      </label>
      <label className="filter-field">
        <span>Module</span>
        <select value={filters.module} onChange={(event) => update('module', event.target.value)}>
          <option value="all">All modules</option>
          {modules.map((module) => (
            <option key={module} value={module}>
              {module}
            </option>
          ))}
        </select>
      </label>
      <label className="filter-field filter-search">
        <span>Search</span>
        <input
          type="search"
          value={filters.search}
          onChange={(event) => update('search', event.target.value)}
          placeholder="Event, detail, frame…"
        />
      </label>
      <label className="filter-field">
        <span>Order</span>
        <select
          value={filters.sort}
          onChange={(event) => updateSort(event.target.value)}
        >
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </select>
      </label>
    </section>
  );
}
