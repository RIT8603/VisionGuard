import type { CameraStatus } from '../../api/types';

const LABEL: Record<CameraStatus, string> = {
  starting: 'Starting',
  live: 'Live',
  stalled: 'Stalled',
  offline: 'Offline',
  failed: 'Failed',
};

const HINT: Record<CameraStatus, string> = {
  starting: 'Opening the camera and loading modules',
  live: 'Frames are flowing',
  stalled: 'No new frame for a few seconds',
  offline: 'The video ended or the stream was lost',
  failed: 'The camera could not be opened or crashed',
};

/** Reuses the dashboard badge colours: live = ok, stalled = warn, failed = crit. */
const TONE: Record<CameraStatus, string> = {
  starting: 'waiting',
  live: 'live',
  stalled: 'stale',
  offline: 'stopped',
  failed: 'offline',
};

export function CameraStatusBadge({ status }: { status: CameraStatus }) {
  return (
    <span className={`badge badge-sm badge-${TONE[status]}`} title={HINT[status]}>
      {LABEL[status]}
    </span>
  );
}
