import { useState } from 'react';
import { ModuleCard } from '../../components/modules/ModuleCard';
import { SelectionSummary } from '../../components/modules/SelectionSummary';
import { CATEGORIES, MODULE_CATALOG } from '../../content/moduleCatalog';
import { usePipeline } from '../../state/PipelineProvider';

/**
 * Review and select capabilities. The backend has no endpoint for changing
 * modules, so selection is local, unsaved state: it lives in this component
 * and is gone on reload. Nothing is written to storage or sent anywhere.
 */
export function ModulesPage() {
  const { status, reachable } = usePipeline();
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set());

  // Modules the connected pipeline is actually running (from /api/status).
  const runningKeys = new Set(
    reachable ? (status.data?.info.modules ?? []).map((m) => m.name) : [],
  );

  const toggle = (id: string) =>
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <>
      <title>Modules · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          {/* h2: the portal header's brand is the page's h1. */}
          <h2 className="page-title">Module selection</h2>
          <p className="page-intro">
            Choose the capabilities your cameras need. Each one shows whether it works today, needs
            setup first, or is still planned.
          </p>
        </header>

        <div className="notice notice-info" role="note">
          <span className="notice-word">Not saved</span>
          <div className="notice-body">
            Modules can’t be switched on from the browser yet: the backend has no endpoint for it.
            Your selection stays in this tab and is lost on reload. To enable modules today, copy the
            generated configuration into <code>configs/config.yaml</code> and restart the backend.
          </div>
        </div>

        <p className="page-meta mono">
          {reachable
            ? `Connected pipeline is running ${runningKeys.size} backend module${runningKeys.size === 1 ? '' : 's'}.`
            : 'Backend not connected — running modules cannot be shown.'}
        </p>

        <div className="modules-layout">
          <div className="modules-groups">
            {CATEGORIES.map((category) => (
              <fieldset key={category} className="module-group">
                <legend>{category}</legend>
                <ul className="module-cards">
                  {MODULE_CATALOG.filter((m) => m.category === category).map((module) => (
                    <ModuleCard
                      key={module.id}
                      module={module}
                      selected={selected.has(module.id)}
                      running={
                        module.backendModules.length > 0 &&
                        module.backendModules.every((key) => runningKeys.has(key))
                      }
                      onToggle={toggle}
                    />
                  ))}
                </ul>
              </fieldset>
            ))}
          </div>

          <aside className="modules-aside" aria-label="Selection summary">
            <SelectionSummary selected={selected} onClear={() => setSelected(new Set())} />
          </aside>
        </div>
      </div>
    </>
  );
}
