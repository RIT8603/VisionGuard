import { useEffect, useRef, useState } from 'react';
import { setModules } from '../../api/endpoints';
import { ApiError } from '../../api/client';
import { ModuleCard } from '../../components/modules/ModuleCard';
import { SelectionSummary } from '../../components/modules/SelectionSummary';
import { CATEGORIES, MODULE_CATALOG } from '../../content/moduleCatalog';
import { usePipeline } from '../../state/PipelineProvider';

type ApplyState = 'idle' | 'applying' | 'applied' | 'error';

/** How long after the last click to send the selection, so a quick run of
 * toggles becomes one backend restart instead of several. */
const APPLY_DEBOUNCE_MS = 400;

/** Backend module keys the current selection needs, dependencies included,
 * in catalogue order. Planned items contribute nothing - there is no module
 * to enable yet. */
function backendKeysFor(selected: ReadonlySet<string>): string[] {
  const keys: string[] = [];
  for (const module of MODULE_CATALOG) {
    if (module.status === 'planned' || !selected.has(module.id)) continue;
    for (const key of module.backendModules) {
      if (!keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

export function ModulesPage() {
  const { status, reachable } = usePipeline();
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set());
  const [apply, setApply] = useState<{ state: ApplyState; message?: string }>({ state: 'idle' });
  // Seeded once the backend's actual running modules are known, so the first
  // render doesn't send an "apply" for an empty selection.
  const seeded = useRef(false);

  // Modules the connected pipeline is actually running (from /api/status).
  const runningKeys = new Set(
    reachable ? (status.data?.info.modules ?? []).map((m) => m.name) : [],
  );

  useEffect(() => {
    if (seeded.current || !reachable || !status.data) return;
    seeded.current = true;
    const ids = MODULE_CATALOG.filter(
      (m) => m.backendModules.length > 0 && m.backendModules.every((key) => runningKeys.has(key)),
    ).map((m) => m.id);
    setSelected(new Set(ids));
    // Only the first time the backend is seen - after that, the operator's
    // own clicks are the source of truth, not whatever is currently running.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reachable, status.data]);

  useEffect(() => {
    if (!seeded.current) return;
    const keys = backendKeysFor(selected);
    const timer = window.setTimeout(() => {
      setApply({ state: 'applying' });
      setModules(keys)
        .then(() => setApply({ state: 'applied' }))
        .catch((error: unknown) => {
          const message = error instanceof ApiError ? error.message : 'Could not reach the backend';
          setApply({ state: 'error', message });
        });
    }, APPLY_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [selected]);

  const toggle = (id: string) =>
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const noticeKind = apply.state === 'error' ? 'warning' : 'info';
  const noticeWord =
    apply.state === 'applying'
      ? 'Applying'
      : apply.state === 'error'
        ? 'Failed'
        : apply.state === 'applied'
          ? 'Applied'
          : 'Live';
  const noticeBody =
    apply.state === 'applying'
      ? 'Sending your selection to the backend. The pipeline restarts with the new modules in a few seconds; the video pauses briefly while it does.'
      : apply.state === 'applied'
        ? 'The running pipeline now matches your selection, and configs/config.yaml was updated to match - it survives a manual restart too.'
        : apply.state === 'error'
          ? `Could not apply the selection: ${apply.message}. The pipeline kept running with its previous modules.`
          : 'Toggling a module here changes the running pipeline directly - no manual config edit or restart needed.';

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

        <div className={`notice notice-${noticeKind}`} role="note">
          <span className="notice-word">{noticeWord}</span>
          <div className="notice-body">{noticeBody}</div>
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
