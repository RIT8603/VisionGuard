import { EventFeed } from '../../components/pipeline/EventFeed';
import { FrameResults } from '../../components/pipeline/FrameResults';
import { LiveStream } from '../../components/pipeline/LiveStream';
import { ModulesPanel } from '../../components/pipeline/ModulesPanel';
import { StatsPanel } from '../../components/pipeline/StatsPanel';
import { StatusPanel } from '../../components/pipeline/StatusPanel';
import { usePipeline } from '../../state/PipelineProvider';

export function LiveMonitoringPage() {
  const { status, feed, visible, reachable, frame, stats } = usePipeline();
  const { data } = status;

  return (
    <>
      <title>Live Monitoring · VisionGuard</title>
      <div className={`layout${reachable ? '' : ' is-dimmed'}`}>
        <div className="column column-main">
          <LiveStream
            viewState={status.viewState}
            active={visible}
            restartCount={status.restartCount}
            resolution={frame?.resolution}
            sourceLabel={data?.info.source_id}
          />
          <StatsPanel stats={stats} eventCount={data?.event_count ?? null} />
          <FrameResults frame={frame} showAll />
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
      </div>
    </>
  );
}
