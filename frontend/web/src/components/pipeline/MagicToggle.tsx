import { useEffect, useState } from 'react';
import { toApiError } from '../../api/client';
import { getViewOptions, setMagicEffect } from '../../api/endpoints';

/**
 * Switches the "magic" movement effect (a glowing trail and sparkles behind
 * each moving person) on or off. It is drawn into the video by the backend for every
 * viewer, so this is a shared setting, not a per-browser one.
 */
export function MagicToggle() {
  const [on, setOn] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getViewOptions(controller.signal)
      .then((view) => setOn(view.magic_effect))
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(toApiError(caught).message);
      });
    return () => controller.abort();
  }, []);

  const toggle = async () => {
    if (on === null) return;
    setBusy(true);
    setError(null);
    try {
      setOn((await setMagicEffect(!on)).magic_effect);
    } catch (caught) {
      setError(toApiError(caught).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="magic-toggle">
      <button
        type="button"
        className={`seg-btn${on ? ' is-active' : ''}`}
        aria-pressed={on ?? false}
        disabled={on === null || busy}
        onClick={() => void toggle()}
      >
        Magic effect {on === null ? '' : on ? 'on' : 'off'}
      </button>
      <span className="field-help">A glowing trail and sparkles that trace how each tracked person moves. People and their boxes stay unchanged.</span>
      {error && <span className="field-problem">{error}</span>}
    </div>
  );
}
