import { EventFeed } from '../../components/pipeline/EventFeed';
import { FrameSummary } from '../../components/pipeline/FrameSummary';
import { LiveStream } from '../../components/pipeline/LiveStream';
import { ModulesPanel } from '../../components/pipeline/ModulesPanel';
import { StatsPanel } from '../../components/pipeline/StatsPanel';
import { StatusPanel } from '../../components/pipeline/StatusPanel';
import { PanelLink } from '../../components/ui/PanelLink';
import { PATHS } from '../../app/paths';
import { usePipeline } from '../../state/PipelineProvider';

export function DashboardPage() {
  const { status, feed, visible, reachable, frame, stats } = usePipeline();
  const { data } = status;

  // Header, navigation and connection notice come from PortalLayout.
  return (
    <>
      <title>Dashboard · VisionGuard</title>

      {/* DOM order is the mobile reading order; wider screens rearrange it in CSS.
          Last known values stay on screen, dimmed, while the backend is away. */}
      <div className={`layout${reachable ? '' : ' is-dimmed'}`}>
        <div className="column column-main">
          <LiveStream
            viewState={status.viewState}
            active={visible}
            restartCount={status.restartCount}
            resolution={frame?.resolution}
            sourceLabel={data?.info.source_id}
          />
          <PanelLink to={PATHS.portal.live}>Live monitoring →</PanelLink>
          <StatsPanel stats={stats} eventCount={data?.event_count ?? null} />
          <FrameSummary frame={frame} />
        </div>
        <div className="column column-side">
          <StatusPanel status={status} frame={frame} />
          <ModulesPanel
            modules={data?.info.modules ?? []}
            latency={stats?.module_latency_ms}
            errors={frame?.errors}
            live={status.viewState === 'live'}
          />
          <PanelLink to={PATHS.portal.modules}>Select →</PanelLink>
          <EventFeed events={feed.events.slice(0, 5)} error={reachable ? feed.error : null} />
          <PanelLink to={PATHS.portal.events}>All events →</PanelLink>
        </div>
      </div>
    </>
  );
}
