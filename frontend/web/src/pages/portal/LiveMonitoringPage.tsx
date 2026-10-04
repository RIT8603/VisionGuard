import { Link } from 'react-router';
import { PATHS } from '../../app/paths';
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
      <div className="notice notice-info" role="note">
        <span className="notice-word">Single source</span>
        <div className="notice-body">
          This page shows the one pipeline the backend runs today. Once the backend can list
          cameras, choosing a camera moves to <Link to={PATHS.portal.cameras}>Cameras</Link>.
        </div>
      </div>
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
