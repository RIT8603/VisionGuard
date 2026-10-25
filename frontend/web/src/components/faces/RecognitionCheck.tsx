import { useState, type ChangeEvent } from 'react';
import { toApiError } from '../../api/client';
import type { AccuracyResponse, IdentifyResponse } from '../../api/faceTypes';
import { getFaceAccuracy, identifyFace } from '../../api/faces';
import { Panel } from '../ui/Panel';

const pct = (value: number | null) => (value === null ? '—' : `${(value * 100).toFixed(1)}%`);
const num = (value: number | null | undefined) => (value === null || value === undefined ? '—' : value.toFixed(3));

/**
 * Check recognition before relying on it: who the live matcher sees in a
 * photo, and a leave-one-out accuracy estimate over the enrolled templates.
 * Neither stores anything; both work in read-only mode.
 */
export function RecognitionCheck({ modelsReady }: { modelsReady: boolean }) {
  const [identity, setIdentity] = useState<IdentifyResponse | null>(null);
  const [accuracy, setAccuracy] = useState<AccuracyResponse | null>(null);
  const [busy, setBusy] = useState<'identify' | 'accuracy' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const testPhoto = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setBusy('identify');
    setError(null);
    setIdentity(null);
    try {
      setIdentity(await identifyFace(file));
    } catch (caught) {
      setError(toApiError(caught).message);
    } finally {
      setBusy(null);
    }
  };

  const estimate = async () => {
    setBusy('accuracy');
    setError(null);
    try {
      setAccuracy(await getFaceAccuracy());
    } catch (caught) {
      setError(toApiError(caught).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Panel title="Check recognition">
      {!modelsReady && <p className="page-meta">Needs the face models - see Registry above.</p>}

      <div className="face-actions">
        <label className={`summary-btn face-upload${!modelsReady || busy ? ' is-disabled' : ''}`}>
          {busy === 'identify' ? 'Checking…' : 'Test a photo'}
          <input
            type="file"
            accept="image/jpeg,image/png"
            disabled={!modelsReady || busy !== null}
            onChange={(event) => void testPhoto(event)}
          />
        </label>
        <button type="button" className="summary-btn summary-btn-quiet" disabled={busy !== null} onClick={() => void estimate()}>
          {busy === 'accuracy' ? 'Estimating…' : 'Estimate accuracy'}
        </button>
      </div>

      {error && <p className="inline-error" role="alert">{error}</p>}

      {identity && (
        <div className="face-result" aria-live="polite">
          <p className={identity.decision === 'unknown' ? 'face-decision face-warn' : 'face-decision face-ok'}>
            {identity.decision === 'unknown' ? 'Unknown' : identity.decision}
          </p>
          <p className="muted">{identity.reason}</p>
          {identity.quality_problems.length > 0 && (
            <p className="face-bad">Photo quality: {identity.quality_problems.join('; ')}</p>
          )}
          {identity.matches.length > 0 && (
            <table className="data-table face-table">
              <thead>
                <tr>
                  <th scope="col">Closest matches</th>
                  <th scope="col">Similarity</th>
                </tr>
              </thead>
              <tbody>
                {identity.matches.map((match) => (
                  <tr key={match.name}>
                    <td>{match.name}</td>
                    <td className={`mono${match.similarity >= identity.threshold ? ' face-ok' : ''}`}>
                      {match.similarity.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="field-help">Threshold {identity.threshold.toFixed(2)}: a name needs at least this similarity and a clear lead.</p>
        </div>
      )}

      {accuracy && (
        <div className="face-result" aria-live="polite">
          <dl className="face-stats">
            <div>
              <dt>People / trials</dt>
              <dd className="mono">
                {accuracy.identities} / {accuracy.trials}
              </dd>
            </div>
            <div>
              <dt>Rank-1 accuracy</dt>
              <dd className="mono">{pct(accuracy.rank1_accuracy)}</dd>
            </div>
            <div>
              <dt>False accept</dt>
              <dd className={`mono${accuracy.false_accept_rate ? ' face-bad' : ''}`}>{pct(accuracy.false_accept_rate)}</dd>
            </div>
            <div>
              <dt>False reject</dt>
              <dd className="mono">{pct(accuracy.false_reject_rate)}</dd>
            </div>
            <div>
              <dt>Genuine (min / mean)</dt>
              <dd className="mono">
                {num(accuracy.genuine?.min)} / {num(accuracy.genuine?.mean)}
              </dd>
            </div>
            <div>
              <dt>Impostor (mean / max)</dt>
              <dd className="mono">
                {num(accuracy.impostor?.mean)} / {num(accuracy.impostor?.max)}
              </dd>
            </div>
            <div>
              <dt>Threshold / suggested</dt>
              <dd className="mono">
                {accuracy.threshold.toFixed(2)} / {num(accuracy.suggested_threshold)}
              </dd>
            </div>
          </dl>
          {accuracy.suggested_threshold !== null &&
            Math.abs(accuracy.suggested_threshold - accuracy.threshold) >= 0.05 && (
              <p className="face-warn">
                The suggested threshold differs from the current one. To change it, edit face_recognition → threshold
                on the Settings page.
              </p>
            )}
          <p className="field-help">{accuracy.note}</p>
        </div>
      )}
    </Panel>
  );
}
