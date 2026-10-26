import { useState } from 'react';
import { toApiError } from '../../api/client';
import { getFaceAccuracy, identifyFace } from '../../api/endpoints';
import type { AccuracyReport, IdentifyResult } from '../../api/types';

const pct = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;

/**
 * Two ways to check recognition before relying on it:
 *
 * - Test a photo: the live matcher's answer for one image, with its top
 *   candidates and why it decided what it did.
 * - Estimate accuracy: every registered template matched against everyone
 *   else's - shows whether people are told apart at the current threshold.
 */
export function RecognitionCheck() {
  const [result, setResult] = useState<IdentifyResult | null>(null);
  const [report, setReport] = useState<AccuracyReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const test = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await identifyFace(file));
    } catch (caught) {
      setResult(null);
      setError(toApiError(caught).message);
    } finally {
      setBusy(false);
    }
  };

  const estimate = async () => {
    setBusy(true);
    setError(null);
    try {
      setReport(await getFaceAccuracy());
    } catch (caught) {
      setError(toApiError(caught).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel recognition-check">
      <header className="panel-head">
        <h2>Check recognition</h2>
      </header>
      <div className="panel-body recognition-check-body">
        <div>
          <h3 className="enroll-side-title">Test a photo</h3>
          <p className="field-help">Upload a photo of one person. Nothing is stored.</p>
          <input
            type="file"
            accept="image/jpeg,image/png"
            disabled={busy}
            onChange={(event) => {
              void test(event.target.files?.[0]);
              event.target.value = '';
            }}
          />
          {result && (
            <div className="recognition-result">
              <p>
                <span className={`faces-badge ${result.decision === 'unknown' ? 'is-warn' : 'is-ok'}`}>
                  {result.decision === 'unknown' ? 'Unknown' : result.decision}
                </span>{' '}
                {result.reason}
              </p>
              {result.matches.length > 0 && (
                <ol className="recognition-matches mono">
                  {result.matches.map((m) => (
                    <li key={m.name}>
                      {m.name} — {m.similarity.toFixed(3)}
                      {m.similarity >= result.threshold ? ' ✓' : ''}
                    </li>
                  ))}
                </ol>
              )}
              {result.quality_problems.length > 0 && (
                <p className="field-help">Photo quality: {result.quality_problems.join('; ')}</p>
              )}
            </div>
          )}
        </div>

        <div>
          <h3 className="enroll-side-title">Estimate accuracy</h3>
          <p className="field-help">Matches every registered capture against everyone else's.</p>
          <button type="button" className="summary-btn" disabled={busy} onClick={() => void estimate()}>
            {busy ? 'Working…' : 'Estimate accuracy'}
          </button>
          {report && (
            <dl className="accuracy-grid">
              <dt>People</dt>
              <dd className="mono">{report.identities}</dd>
              <dt>Correct (rank-1)</dt>
              <dd className="mono">{pct(report.rank1_accuracy)}</dd>
              <dt>False accept rate</dt>
              <dd className="mono">{pct(report.false_accept_rate)}</dd>
              <dt>False reject rate</dt>
              <dd className="mono">{pct(report.false_reject_rate)}</dd>
              <dt>Threshold</dt>
              <dd className="mono">
                {report.threshold.toFixed(3)}
                {report.suggested_threshold !== null && ` (suggested ${report.suggested_threshold.toFixed(3)})`}
              </dd>
              <dt>Same person</dt>
              <dd className="mono">{report.genuine ? `${report.genuine.min} – ${report.genuine.max}` : '—'}</dd>
              <dt>Different people</dt>
              <dd className="mono">{report.impostor ? `${report.impostor.min} – ${report.impostor.max}` : '—'}</dd>
            </dl>
          )}
          {report && <p className="field-help">{report.note}</p>}
        </div>
        {error && <p className="inline-error">{error}</p>}
      </div>
    </section>
  );
}
