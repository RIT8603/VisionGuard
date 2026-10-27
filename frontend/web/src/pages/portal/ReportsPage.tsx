import { useEffect, useState } from 'react';
import { toApiError } from '../../api/client';
import {
  attendanceCsvUrl,
  eventLogCsvUrl,
  getEventLog,
  getLiveReport,
  snapshotImageUrl,
  type EventLogFilters,
} from '../../api/endpoints';
import type { EventLogPage, LiveReport, LoggedEvent } from '../../api/types';
import { usePageVisible } from '../../hooks/usePageVisible';
import { fmtTime } from '../../lib/format';

const LIVE_POLL_MS = 2000;
const LOG_POLL_MS = 5000;
const PAGE_SIZE = 50;

/** Event-type filters offered on the log; each maps to one or more event names. */
const TYPE_FILTERS: { value: string; label: string; events: string }[] = [
  { value: '', label: 'All events', events: '' },
  { value: 'gate', label: 'Entries & exits', events: 'gate_entry,gate_exit' },
  { value: 'unknown', label: 'Unknown people', events: 'unknown_person' },
  { value: 'spoof', label: 'Spoof attempts', events: 'spoof_suspected' },
  { value: 'attendance', label: 'Attendance', events: 'attendance_check_in,attendance_check_out' },
  { value: 'weapon', label: 'Weapons', events: 'weapon_detected,armed_person' },
];

const EVENT_LABEL: Record<string, string> = {
  gate_entry: 'Entered',
  gate_exit: 'Exited',
  unknown_person: 'Unknown person',
  spoof_suspected: 'Spoof suspected',
  attendance_check_in: 'Check-in',
  attendance_check_out: 'Check-out',
  weapon_detected: 'Weapon',
  armed_person: 'Armed person',
  fire_detected: 'Fire',
  smoke_detected: 'Smoke',
  ppe_violation: 'PPE violation',
};

const today = () => new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD, local

export function ReportsPage() {
  const visible = usePageVisible();
  const [live, setLive] = useState<LiveReport | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);

  const [type, setType] = useState('gate');
  const [camera, setCamera] = useState('');
  const [identity, setIdentity] = useState('');
  const [day, setDay] = useState(today());
  const [page, setPage] = useState(0);
  const [log, setLog] = useState<EventLogPage | null>(null);
  const [logError, setLogError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const filters: EventLogFilters = {
    camera: camera || undefined,
    event: TYPE_FILTERS.find((f) => f.value === type)?.events || undefined,
    identity: identity.trim() || undefined,
    day: day || undefined,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  const filterKey = JSON.stringify(filters);

  // Live counts: polled while the page is visible.
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const tick = async () => {
      try {
        const next = await getLiveReport(controller.signal);
        if (controller.signal.aborted) return;
        setLive(next);
        setLiveError(null);
      } catch (caught) {
        if (controller.signal.aborted) return;
        setLiveError(toApiError(caught).message);
      }
      // Never reschedule after cleanup: a late reply must not start a loop of its own.
      if (!controller.signal.aborted) timer = window.setTimeout(tick, LIVE_POLL_MS);
    };
    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [visible]);

  // The log: reloaded when the filters change, and every few seconds on page 1.
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const load = async () => {
      try {
        const next = await getEventLog(JSON.parse(filterKey) as EventLogFilters, controller.signal);
        // The filters may have changed while this was in flight: drop the stale reply.
        if (controller.signal.aborted) return;
        setLog(next);
        setLogError(null);
      } catch (caught) {
        if (controller.signal.aborted) return;
        setLogError(toApiError(caught).message);
      }
      if (page === 0 && !controller.signal.aborted) timer = window.setTimeout(load, LOG_POLL_MS);
    };
    void load();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [filterKey, visible, page, nonce]);

  const todayStats = live?.today;
  const pages = log ? Math.max(1, Math.ceil(log.total / PAGE_SIZE)) : 1;
  const resetPage = <T,>(set: (value: T) => void) => (value: T) => {
    set(value);
    setPage(0);
  };

  return (
    <>
      <title>Reports · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Reports</h2>
          <p className="page-intro">
            Who is here right now, every entry and exit through the gate, unknown people with their
            snapshot, and spoof attempts. Everything is kept in the event log and can be exported as CSV.
          </p>
        </header>

        {liveError && <p className="inline-error">Live counts unavailable: {liveError}</p>}

        <dl className="faces-strip">
          <Stat label="People in view" value={live?.people_in_view} />
          <Stat label="Entered today" value={todayStats?.entered} />
          <Stat label="Exited today" value={todayStats?.exited} />
          <Stat label="Inside now" value={todayStats?.inside} />
          <Stat label="Unknown today" value={todayStats?.by_type.unknown_person ?? (todayStats ? 0 : undefined)} />
        </dl>

        <div className="reports-grid">
          <section className="panel">
            <header className="panel-head">
              <h2>Cameras</h2>
            </header>
            <div className="panel-body">
              {(live?.cameras ?? []).length === 0 ? (
                <p className="empty">No cameras.</p>
              ) : (
                <ul className="report-cameras">
                  {live?.cameras.map((cam) => (
                    <li key={cam.id}>
                      <div className="report-camera-head">
                        <strong>{cam.name}</strong>
                        <span className={`faces-badge ${cam.status === 'live' ? 'is-ok' : 'is-warn'}`}>
                          {cam.status}
                        </span>
                      </div>
                      <div className="report-camera-stats mono">
                        {cam.people_in_view} in view · {cam.recognised.length} recognised ·{' '}
                        {cam.unknown_faces} unknown{cam.spoof_faces ? ` · ${cam.spoof_faces} spoof` : ''}
                      </div>
                      {cam.recognised.length > 0 && (
                        <div className="report-names">{cam.recognised.join(', ')}</div>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          <section className="panel">
            <header className="panel-head">
              <h2>Inside now</h2>
              <span className="panel-aside">{todayStats?.people_inside.length ?? 0} named</span>
            </header>
            <div className="panel-body">
              {(todayStats?.people_inside ?? []).length === 0 ? (
                <p className="empty">
                  No recognised person has entered without leaving today. People entering without a
                  recognised face are counted in "Inside now" but cannot be named.
                </p>
              ) : (
                <ul className="report-inside">
                  {todayStats?.people_inside.map((p) => (
                    <li key={p.identity}>
                      <strong>{p.identity}</strong>
                      <span className="mono"> since {fmtTime(p.since)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>

        <section className="panel report-log">
          <header className="panel-head">
            <h2>Event log</h2>
            <span className="panel-aside">{log ? `${log.total} events` : ''}</span>
          </header>
          <div className="panel-body">
            <div className="report-filters">
              <div className="field">
                <label htmlFor="log-type">Show</label>
                <select id="log-type" value={type} onChange={(e) => resetPage(setType)(e.target.value)}>
                  {TYPE_FILTERS.map((f) => (
                    <option key={f.value} value={f.value}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="log-camera">Camera</label>
                <select id="log-camera" value={camera} onChange={(e) => resetPage(setCamera)(e.target.value)}>
                  <option value="">All cameras</option>
                  {(live?.cameras ?? []).map((cam) => (
                    <option key={cam.id} value={cam.id}>
                      {cam.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="log-day">Day</label>
                <input id="log-day" type="date" value={day} onChange={(e) => resetPage(setDay)(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="log-identity">Person</label>
                <input
                  id="log-identity"
                  type="text"
                  placeholder="Exact name"
                  value={identity}
                  onChange={(e) => resetPage(setIdentity)(e.target.value)}
                />
              </div>
              <div className="report-exports">
                <a className="summary-btn" href={eventLogCsvUrl(filters)} download>
                  Export CSV
                </a>
                <a className="summary-btn summary-btn-quiet report-export-quiet" href={attendanceCsvUrl(day)} download>
                  Attendance CSV
                </a>
                <button type="button" className="link-btn" onClick={() => setNonce((n) => n + 1)}>
                  Refresh
                </button>
              </div>
            </div>

            {logError && <p className="inline-error">Event log unavailable: {logError}</p>}

            {log && log.events.length === 0 ? (
              <p className="empty">No events match these filters.</p>
            ) : (
              <div className="faces-table-wrap">
                <table className="faces-table report-table">
                  <thead>
                    <tr>
                      <th scope="col">Time</th>
                      <th scope="col">Camera</th>
                      <th scope="col">Event</th>
                      <th scope="col">Person</th>
                      <th scope="col">Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {log?.events.map((event) => <LogRow key={event.id} event={event} />)}
                  </tbody>
                </table>
              </div>
            )}

            {pages > 1 && (
              <div className="report-pager">
                <button type="button" className="link-btn" disabled={page === 0} onClick={() => setPage(page - 1)}>
                  ← Newer
                </button>
                <span className="mono">
                  page {page + 1} of {pages}
                </span>
                <button
                  type="button"
                  className="link-btn"
                  disabled={page + 1 >= pages}
                  onClick={() => setPage(page + 1)}
                >
                  Older →
                </button>
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  );
}

function Stat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="faces-strip-item">
      <dt>{label}</dt>
      <dd className="mono">{value ?? '—'}</dd>
    </div>
  );
}

function LogRow({ event }: { event: LoggedEvent }) {
  const data = event.data;
  const snapshot = typeof data.snapshot === 'string' ? data.snapshot : null;
  const frame = typeof data.frame_snapshot === 'string' ? data.frame_snapshot : null;
  const details: string[] = [];
  if (typeof data.gate === 'string') details.push(data.gate);
  if (typeof data.inside === 'number') details.push(`inside ${data.inside}`);
  if (typeof data.closest_similarity === 'number') details.push(`closest match ${data.closest_similarity.toFixed(2)}`);
  // Weapon / fire / smoke / PPE events: what was seen, and how sure the model was.
  if (Array.isArray(data.classes) && data.classes.length) details.push(data.classes.join(', '));
  if (Array.isArray(data.missing) && data.missing.length) details.push(`missing ${data.missing.join(', ')}`);
  if (typeof data.confidence === 'number') details.push(`${Math.round(data.confidence * 100)}%`);
  if (event.track_id !== null) details.push(`track #${event.track_id}`);
  const tone =
    event.event === 'gate_entry' ? 'is-ok' : event.event === 'gate_exit' ? '' : event.severity !== 'info' ? 'is-warn' : '';

  return (
    <tr>
      <td className="mono">{fmtTime(event.timestamp)}</td>
      <td>{event.camera}</td>
      <td>
        <span className={`faces-badge ${tone}`}>{EVENT_LABEL[event.event] ?? event.event}</span>
      </td>
      <td>{event.identity && event.identity !== 'unknown' ? <strong>{event.identity}</strong> : '—'}</td>
      <td>
        <div className="report-details">
          {snapshot && (
            <a href={snapshotImageUrl(frame ?? snapshot)} target="_blank" rel="noreferrer" title="Open the full frame">
              <img className="report-snapshot" src={snapshotImageUrl(snapshot)} alt={`${EVENT_LABEL[event.event] ?? event.event} snapshot`} loading="lazy" />
            </a>
          )}
          <span>{details.join(' · ')}</span>
        </div>
      </td>
    </tr>
  );
}
