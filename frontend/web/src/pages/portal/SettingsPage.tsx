import { useCallback, useEffect, useState } from 'react';
import { toApiError } from '../../api/client';
import { getSettings, setSettings } from '../../api/endpoints';
import type { ModuleSettings, SettingValue, SettingsResponse } from '../../api/types';
import { usePageVisible } from '../../hooks/usePageVisible';

/** While a change is being applied, re-read the settings this often. */
const APPLY_POLL_MS = 1000;
const APPLY_POLL_LIMIT = 20;

type Draft = Record<string, string | boolean>;

function toDraft(settings: Record<string, SettingValue>): Draft {
  const draft: Draft = {};
  for (const [key, { value }] of Object.entries(settings)) {
    draft[key] = typeof value === 'boolean' ? value : String(value);
  }
  return draft;
}

/** The changed values, parsed; or an error message for the first bad one. */
function changes(
  settings: Record<string, SettingValue>,
  draft: Draft,
): { values: Record<string, boolean | number | string> } | { error: string } {
  const values: Record<string, boolean | number | string> = {};
  for (const [key, { value, kind }] of Object.entries(settings)) {
    const raw = draft[key];
    if (kind === 'bool' || kind === 'severity') {
      if (raw !== value) values[key] = raw as boolean | string;
      continue;
    }
    const parsed = Number(raw);
    if (String(raw).trim() === '' || !Number.isFinite(parsed) || parsed < 0) {
      return { error: `${key}: enter a number of 0 or more` };
    }
    if (kind === 'int' && !Number.isInteger(parsed)) return { error: `${key}: enter a whole number` };
    if (parsed !== value) values[key] = parsed;
  }
  return { values };
}

export function SettingsPage() {
  const visible = usePageVisible();
  const [data, setData] = useState<SettingsResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const body = await getSettings();
      setData(body);
      setLoadError(null);
      return body;
    } catch (caught) {
      setLoadError(toApiError(caught).message);
      return null;
    }
  }, []);

  useEffect(() => {
    if (visible) void load();
  }, [visible, load]);

  const modules = Object.entries(data?.modules ?? {}).sort(
    ([a, x], [b, y]) => Number(y.enabled) - Number(x.enabled) || a.localeCompare(b),
  );

  return (
    <>
      <title>Settings · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          {/* h2: the portal header's brand is the page's h1. */}
          <h2 className="page-title">Module settings</h2>
          <p className="page-intro">
            Tune thresholds, timers and severities without editing config.yaml. Saving restarts
            the modules for a few seconds and writes the new value into config.yaml. Model files,
            paths, sources and zones are only editable in the file.
          </p>
        </header>

        {loadError && <p className="inline-error">Settings unavailable: {loadError}</p>}
        {data && !data.editable && (
          <p className="page-meta mono">Read-only for this run (dashboard.allow_settings_edit: false).</p>
        )}
        {data?.apply_error && (
          <p className="inline-error">Last change was not applied: {data.apply_error}</p>
        )}

        {modules.length === 0 && data && <p className="empty">No module has editable settings.</p>}
        <div className="settings-grid">
          {modules.map(([name, module]) => (
            <ModuleSettingsCard
              key={name}
              name={name}
              module={module}
              editable={data?.editable ?? false}
              reload={load}
            />
          ))}
        </div>
      </div>
    </>
  );
}

interface CardProps {
  name: string;
  module: ModuleSettings;
  editable: boolean;
  reload: () => Promise<SettingsResponse | null>;
}

function ModuleSettingsCard({ name, module, editable, reload }: CardProps) {
  const [draft, setDraft] = useState<Draft>(() => toDraft(module.settings));
  const [message, setMessage] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null);
  const [saving, setSaving] = useState(false);

  // A change applied elsewhere (another tab, a reload) resets an untouched card.
  const serverKey = JSON.stringify(module.settings);
  useEffect(() => {
    if (!saving) setDraft(toDraft(module.settings));
  }, [serverKey]);

  const parsed = changes(module.settings, draft);
  const dirty = 'values' in parsed ? Object.keys(parsed.values).length > 0 : true;

  const save = async () => {
    if ('error' in parsed) {
      setMessage({ kind: 'error', text: parsed.error });
      return;
    }
    setSaving(true);
    setMessage(null);
    try {
      await setSettings(name, parsed.values);
      // Wait for the restart: the new value shows up, or apply_error does.
      for (let i = 0; i < APPLY_POLL_LIMIT; i += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, APPLY_POLL_MS));
        const body = await reload();
        if (!body) continue;
        if (body.apply_error) {
          setMessage({ kind: 'error', text: body.apply_error });
          return;
        }
        const now = body.modules[name]?.settings ?? {};
        if (Object.entries(parsed.values).every(([key, value]) => now[key]?.value === value)) {
          setMessage({ kind: 'ok', text: 'Applied and saved.' });
          return;
        }
      }
      setMessage({ kind: 'error', text: 'Still applying - check back in a moment.' });
    } catch (caught) {
      setMessage({ kind: 'error', text: toApiError(caught).message });
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="panel settings-card">
      <header className="panel-head">
        <h2 className="mono">{name}</h2>
        <span className="panel-aside">{module.enabled ? 'enabled' : 'off'}</span>
      </header>
      <div className="panel-body">
        <dl className="settings-fields">
          {Object.entries(module.settings).map(([key, { kind }]) => {
            const id = `setting-${name}-${key}`;
            return (
              <div key={key} className="settings-field">
                <dt>
                  <label htmlFor={id}>{key}</label>
                </dt>
                <dd>
                  {kind === 'bool' ? (
                    <input
                      id={id}
                      type="checkbox"
                      checked={draft[key] === true}
                      disabled={!editable || saving}
                      onChange={(e) => setDraft({ ...draft, [key]: e.target.checked })}
                    />
                  ) : kind === 'severity' ? (
                    <select
                      id={id}
                      value={String(draft[key])}
                      disabled={!editable || saving}
                      onChange={(e) => setDraft({ ...draft, [key]: e.target.value })}
                    >
                      <option value="info">info</option>
                      <option value="warning">warning</option>
                      <option value="critical">critical</option>
                    </select>
                  ) : (
                    <input
                      id={id}
                      type="number"
                      min={0}
                      step={kind === 'int' ? 1 : 'any'}
                      inputMode={kind === 'int' ? 'numeric' : 'decimal'}
                      value={String(draft[key])}
                      disabled={!editable || saving}
                      onChange={(e) => setDraft({ ...draft, [key]: e.target.value })}
                    />
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
        {editable && (
          <div className="settings-actions">
            <button
              type="button"
              className="summary-btn"
              disabled={!dirty || saving}
              onClick={() => void save()}
            >
              {saving ? 'Applying...' : 'Save'}
            </button>
            {dirty && !saving && (
              <button
                type="button"
                className="summary-btn summary-btn-quiet"
                onClick={() => {
                  setDraft(toDraft(module.settings));
                  setMessage(null);
                }}
              >
                Reset
              </button>
            )}
          </div>
        )}
        {message && (
          <p className={message.kind === 'ok' ? 'settings-ok' : 'inline-error'} role="status">
            {message.text}
          </p>
        )}
      </div>
    </section>
  );
}
