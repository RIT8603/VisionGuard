import type { BBox, EventData, EventSubject, Severity } from '../api/types';

export const DASH = '–';

export function fmtNumber(value: number | undefined, digits = 0): string {
  return value !== undefined && Number.isFinite(value) ? value.toFixed(digits) : DASH;
}

export function fmtMs(value: number | undefined, digits = 1): string {
  const text = fmtNumber(value, digits);
  return text === DASH ? text : `${text} ms`;
}

export function fmtPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** 42 -> "42s", 185 -> "3m 05s", 3720 -> "1h 02m" */
export function fmtDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = s % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
  return `${seconds}s`;
}

export function fmtTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleTimeString();
}

export function fmtBox([x1, y1, x2, y2]: BBox): string {
  const r = Math.round;
  return `${r(x1)}, ${r(y1)} → ${r(x2)}, ${r(y2)}`;
}

/** Unknown severities from a future backend render as info rather than unstyled. */
export function severityOf(value: string): Severity {
  return value === 'warning' || value === 'critical' ? value : 'info';
}

/** "person #3", "alice", "2 × face" */
export function describeSubject(subject: EventSubject | undefined): string {
  if (!subject) return '';
  if (typeof subject.identity === 'string') return subject.identity;
  const label = typeof subject.class === 'string' ? subject.class : '';
  if (typeof subject.track_id === 'number') return `${label} #${subject.track_id}`.trim();
  if (typeof subject.count === 'number') return `${subject.count} × ${label}`.trim();
  return label;
}

/** The fields of the known event types worth a line in the feed. */
export function describeData(data: EventData | undefined): string {
  if (!data) return '';
  const parts: string[] = [];
  if (typeof data.zone === 'string') parts.push(`zone ${data.zone}`);
  if (typeof data.dwell_seconds === 'number') parts.push(`dwell ${fmtNumber(data.dwell_seconds, 1)} s`);
  if (typeof data.duration_seconds === 'number') parts.push(`present ${fmtDuration(data.duration_seconds)}`);
  if (typeof data.score === 'number') parts.push(`score ${fmtNumber(data.score, 2)}`);
  if (typeof data.closest_similarity === 'number') {
    parts.push(`closest match ${fmtNumber(data.closest_similarity, 2)}`);
  }
  if (data.reason === 'track_lost') parts.push('track lost');
  return parts.join(' · ');
}
