import { useState } from 'react';
import { MODULE_CATALOG, type CatalogModule } from '../../content/moduleCatalog';
import { Panel } from '../ui/Panel';

interface SelectionSummaryProps {
  selected: ReadonlySet<string>;
  onClear: () => void;
}

/** Backend module keys for the selection, dependencies included, in catalogue order. */
function backendKeysFor(modules: CatalogModule[]): string[] {
  const keys: string[] = [];
  for (const module of modules) {
    for (const key of module.backendModules) {
      if (!keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

function toConfigYaml(keys: string[], setupKeys: ReadonlySet<string>): string {
  const lines = ['modules:'];
  for (const key of keys) {
    lines.push(`  ${key}:`, '    enabled: true');
    if (setupKeys.has(key)) lines.push('    # requires setup: model path (see README)');
  }
  return lines.join('\n');
}

export function SelectionSummary({ selected, onClear }: SelectionSummaryProps) {
  const [copied, setCopied] = useState(false);

  const chosen = MODULE_CATALOG.filter((m) => selected.has(m.id));
  const enableable = chosen.filter((m) => m.status !== 'planned');
  const planned = chosen.filter((m) => m.status === 'planned');
  const needsSetup = chosen.filter((m) => m.status === 'setup');

  const keys = backendKeysFor(enableable);
  const setupKeys = new Set(backendKeysFor(needsSetup));
  const yaml = toConfigYaml(keys, setupKeys);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(yaml);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <Panel
      title="Your selection"
      className="selection-summary"
      aside={<span className="mono">{chosen.length} selected</span>}
    >
      {chosen.length === 0 ? (
        <p className="empty">Select modules to see what enabling them involves.</p>
      ) : (
        <>
          {keys.length > 0 && (
            <section className="summary-block">
              <h3>Applied to the backend</h3>
              <p className="summary-help">
                This selection is sent to the running backend automatically, and written into{' '}
                <code>configs/config.yaml</code> so it survives a manual restart too. Required
                dependencies are included below for reference, or to set up another run by hand.
              </p>
              <pre className="summary-yaml mono">{yaml}</pre>
              <button type="button" className="summary-btn" onClick={copy}>
                {copied ? 'Copied' : 'Copy configuration'}
              </button>
            </section>
          )}

          {needsSetup.length > 0 && (
            <section className="summary-block">
              <h3>Setup required first</h3>
              <ul className="summary-list">
                {needsSetup.map((m) => (
                  <li key={m.id}>
                    <strong>{m.name}:</strong> {m.note}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {planned.length > 0 && (
            <section className="summary-block">
              <h3>Planned — cannot be enabled</h3>
              <p className="summary-help">
                Recorded here as interest only. These do not exist in the backend yet.
              </p>
              <ul className="summary-list">
                {planned.map((m) => (
                  <li key={m.id}>{m.name}</li>
                ))}
              </ul>
            </section>
          )}

          <button type="button" className="summary-btn summary-btn-quiet" onClick={onClear}>
            Clear selection
          </button>
        </>
      )}
    </Panel>
  );
}
