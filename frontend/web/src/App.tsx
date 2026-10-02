import { hasFrame, hasStats } from './api/guards';
import { ConnectionNotice } from './components/ConnectionNotice';
import { EventFeed } from './components/EventFeed';
import { FrameResults } from './components/FrameResults';
import { Header } from './components/Header';
import { LiveStream } from './components/LiveStream';
import { ModulesPanel } from './components/ModulesPanel';
import { StatsPanel } from './components/StatsPanel';
import { StatusPanel } from './components/StatusPanel';
import { useEventFeed } from './hooks/useEventFeed';
import { usePageVisible } from './hooks/usePageVisible';
import { usePipelineStatus } from './hooks/usePipelineStatus';
import { isReachable } from './state/viewState';

export function App() {
  const visible = usePageVisible();
  const status = usePipelineStatus(visible);
  const reachable = isReachable(status.viewState);

  const { data } = status;
  const frame = data && hasFrame(data.frame) ? data.frame : null;
  const stats = data && hasStats(data.stats) ? data.stats : null;

  const feed = useEventFeed({
    active: visible && reachable,
    restartCount: status.restartCount,
    eventCount: data?.event_count ?? null,
  });

  return (
    <div className="app">
      <Header info={data?.info} viewState={status.viewState} uptimeSec={data?.uptime_sec} />
      <ConnectionNotice status={status} />

      {/* DOM order is the mobile reading order: feed, telemetry, frame,
          source, modules, events. Wider screens rearrange it in CSS.
          Last known values stay on screen, dimmed, while the backend is away. */}
      <main className={`layout${reachable ? '' : ' is-dimmed'}`}>
        <div className="column column-main">
          <LiveStream
            viewState={status.viewState}
            active={visible}
            restartCount={status.restartCount}
            resolution={frame?.resolution}
            sourceLabel={data?.info.source_id}
          />
          <StatsPanel stats={stats} eventCount={data?.event_count ?? null} />
          <FrameResults frame={frame} />
        </div>
        <div className="column column-side">
          <StatusPanel status={status} frame={frame} />
          <ModulesPanel
            modules={data?.info.modules ?? []}
            latency={stats?.module_latency_ms}
            errors={frame?.errors}
            live={status.viewState === 'live'}
          />
          <EventFeed events={feed.events} error={reachable ? feed.error : null} />
        </div>
      </main>
    </div>
  );
}
