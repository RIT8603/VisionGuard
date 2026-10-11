import type { ReactNode } from 'react';
import type { ModuleInfo } from '../../api/types';
import { DASH, fmtNumber } from '../../lib/format';
import { Panel } from '../ui/Panel';

interface ModulesPanelProps {
  modules: ModuleInfo[];
  latency: Record<string, number> | undefined;
  /** Modules that raised on the current frame. */
  errors: Record<string, string> | undefined;
  /** Frames are flowing; without that, "no error" says nothing about health. */
  live: boolean;
  /** Rendered in the panel header after the count, e.g. a link. */
  action?: ReactNode;
}

type Health = 'ok' | 'err' | 'unknown';

const HEALTH_LABEL: Record<Health, string> = { ok: 'OK', err: 'ERR', unknown: DASH };

export function ModulesPanel({ modules, latency, errors, live, action }: ModulesPanelProps) {
  const failing = modules.filter((m) => errors?.[m.name]).length;

  return (
    <Panel
      title="Modules"
      aside={
        <>
          {failing > 0 ? (
            <span className="text-crit">{failing} failing</span>
          ) : (
            `${modules.length} loaded`
          )}
          {action}
        </>
      }
    >
      {modules.length === 0 ? (
        <p className="empty">No modules enabled.</p>
      ) : (
        <ul className="modules">
          {modules.map((module) => {
            const error = errors?.[module.name];
            const health: Health = error ? 'err' : live ? 'ok' : 'unknown';
            const ms = latency?.[module.name];
            return (
              <li key={module.name} className="module">
                <span className={`module-health health-${health}`}>{HEALTH_LABEL[health]}</span>
                <span className="module-name">{module.name}</span>
                <span className={`module-type type-${module.type}`}>{module.type}</span>
                <span className="module-ms mono">
                  {ms === undefined ? DASH : fmtNumber(ms, 2)}
                  <span className="unit">ms</span>
                </span>
                {error && <div className="module-error mono">{error}</div>}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
