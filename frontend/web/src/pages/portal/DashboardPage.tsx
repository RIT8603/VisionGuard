import { CameraStrip } from '../../components/pipeline/CameraStrip';
import { EventFeed } from '../../components/pipeline/EventFeed';
import { FrameResults } from '../../components/pipeline/FrameResults';
import { LiveStream } from '../../components/pipeline/LiveStream';
import { ModulesPanel } from '../../components/pipeline/ModulesPanel';
import { StatsPanel } from '../../components/pipeline/StatsPanel';
import { StatusPanel } from '../../components/pipeline/StatusPanel';
import { usePipeline } from '../../state/PipelineProvider';

export function DashboardPage() {
  const { status, feed, visible, reachable, frame, stats, cameras, camera, cameraParam, selectCamera } =
    usePipeline();
  const { data } = status;
  const cameraList = cameras.data?.cameras ?? [];
  const selectedCamera = cameraList.find((c) => c.id === camera);

  // Header, navigation and connection notice come from PortalLayout.
  return (
    <>
      <title>Dashboard · VisionGuard</title>

      {/* DOM order is the mobile reading order: feed, telemetry, frame,
          source, modules, events. Wider screens rearrange it in CSS.
          Last known values stay on screen, dimmed, while the backend is away. */}
      {cameraList.length > 1 && (
        <CameraStrip
          cameras={cameraList}
          selected={camera}
          onSelect={selectCamera}
          active={visible}
        />
      )}

      <div className={`layout${reachable ? '' : ' is-dimmed'}`}>
        <div className="column column-main">
          <LiveStream
            viewState={status.viewState}
            active={visible}
            restartCount={status.restartCount}
            resolution={frame?.resolution}
            sourceLabel={selectedCamera?.name ?? data?.info.source_id}
            camera={cameraParam}
            cameraStatus={selectedCamera?.status}
            cameraError={selectedCamera?.error}
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
      </div>
    </>
  );
}
