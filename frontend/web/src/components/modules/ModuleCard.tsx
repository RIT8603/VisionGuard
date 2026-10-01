import { PLAN_SOURCE_LABEL, STATUS_LABEL, type CatalogModule } from '../../content/moduleCatalog';

interface ModuleCardProps {
  module: CatalogModule;
  selected: boolean;
  /** Every backend module it needs is running in the connected pipeline. */
  running: boolean;
  onToggle: (id: string) => void;
}

/** One selectable capability. The whole card is the checkbox's label. */
export function ModuleCard({ module, selected, running, onToggle }: ModuleCardProps) {
  const inputId = `module-${module.id}`;
  const classes = ['module-card', `module-card-${module.status}`, selected && 'is-selected']
    .filter(Boolean)
    .join(' ');

  return (
    <li className={classes}>
      <label htmlFor={inputId} className="module-card-label">
        <span className="module-card-top">
          <input
            id={inputId}
            type="checkbox"
            checked={selected}
            onChange={() => onToggle(module.id)}
          />
          <span className="module-card-name">{module.name}</span>
          <span className={`module-status module-status-${module.status}`}>
            {STATUS_LABEL[module.status]}
          </span>
        </span>

        {running && <span className="module-running">Running now</span>}
        {module.planSource && (
          <span className="module-plan-source">{PLAN_SOURCE_LABEL[module.planSource]}</span>
        )}

        <span className="module-card-summary">{module.summary}</span>
        {module.note && <span className="module-card-note">{module.note}</span>}

        <span className="module-card-keys mono">
          {module.backendModules.length > 0 ? module.backendModules.join(' · ') : 'no backend module yet'}
        </span>
      </label>
    </li>
  );
}
