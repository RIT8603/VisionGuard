import { VIEW_STATE_LABEL, type ViewState } from '../../state/viewState';

/** Single words, so the state reads at a glance. */
const SHORT_LABEL: Record<ViewState, string> = {
  loading: 'Connecting',
  waiting: 'Waiting',
  live: 'Live',
  stale: 'Stale',
  stopped: 'Stopped',
  offline: 'Offline',
};

interface StateBadgeProps {
  state: ViewState;
  size?: 'sm' | 'lg';
}

export function StateBadge({ state, size = 'sm' }: StateBadgeProps) {
  return (
    <span className={`badge badge-${size} badge-${state}`} title={VIEW_STATE_LABEL[state]}>
      {SHORT_LABEL[state]}
    </span>
  );
}
