import type { Severity } from '../../api/types';
import {
  DEFAULT_FILTER,
  isDefaultFilter,
  SEVERITIES,
  type EventFilter,
  type EventOrder,
} from '../../lib/eventFilters';

const SEVERITY_LABEL: Record<Severity, string> = {
  info: 'Info',
  warning: 'Warning',
  critical: 'Critical',
};

interface EventFiltersProps {
  value: EventFilter;
  onChange: (next: EventFilter) => void;
  /** Modules that have produced events so far. */
  modules: readonly string[];
  counts: Record<Severity, number>;
}

export function EventFilters({ value, onChange, modules, counts }: EventFiltersProps) {
  const toggleSeverity = (severity: Severity) => {
    const next = new Set(value.severities);
    if (next.has(severity)) next.delete(severity);
    else next.add(severity);
    onChange({ ...value, severities: next });
  };

  return (
    <div className="filter-bar" role="group" aria-label="Filter events">
      <div className="filter-field">
        <span className="filter-label" id="severity-filter-label">
          Severity
        </span>
        <div className="filter-chips" role="group" aria-labelledby="severity-filter-label">
          {SEVERITIES.map((severity) => (
            <button
              key={severity}
              type="button"
              className={`filter-chip chip-${severity}`}
              aria-pressed={value.severities.has(severity)}
              onClick={() => toggleSeverity(severity)}
            >
              {SEVERITY_LABEL[severity]}
              <span className="mono filter-chip-count">{counts[severity]}</span>
            </button>
          ))}
        </div>
      </div>

      <label className="filter-field">
        <span className="filter-label">Module</span>
        <select
          className="filter-input"
          value={value.module ?? ''}
          onChange={(e) => onChange({ ...value, module: e.target.value || null })}
        >
          <option value="">All modules</option>
          {modules.map((module) => (
            <option key={module} value={module}>
              {module}
            </option>
          ))}
        </select>
      </label>

      <label className="filter-field filter-field-grow">
        <span className="filter-label">Search</span>
        <input
          type="search"
          className="filter-input"
          placeholder="Event, subject, zone…"
          value={value.query}
          onChange={(e) => onChange({ ...value, query: e.target.value })}
        />
      </label>

      <label className="filter-field">
        <span className="filter-label">Order</span>
        <select
          className="filter-input"
          value={value.order}
          onChange={(e) => onChange({ ...value, order: e.target.value as EventOrder })}
        >
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </select>
      </label>

      {!isDefaultFilter(value) && (
        <button type="button" className="filter-reset" onClick={() => onChange(DEFAULT_FILTER)}>
          Reset
        </button>
      )}
    </div>
  );
}
