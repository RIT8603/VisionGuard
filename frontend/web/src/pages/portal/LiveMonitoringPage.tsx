import { FrameResults } from '../../components/pipeline/FrameResults';
import { LiveStream } from '../../components/pipeline/LiveStream';
import { MagicToggle } from '../../components/pipeline/MagicToggle';
import { ModulesPanel } from '../../components/pipeline/ModulesPanel';
import { StatsPanel } from '../../components/pipeline/StatsPanel';
import { StatusPanel } from '../../components/pipeline/StatusPanel';
import { usePipeline } from '../../state/PipelineProvider';

/**
 * The camera at full detail: stream, every detection and track in the current
 * frame, source state and timing. Same data as the dashboard (one polling loop
 * via PipelineProvider); only one stream is open because only one page is.
 */
export function LiveMonitoringPage() {
  const { status, visible, reachable, frame, stats, cameras, camera, cameraParam } = usePipeline();
  const { data } = status;
  const selectedCamera = cameras.data?.cameras.find((c) => c.id === camera);

  return (
    <>
      <title>Live Monitoring · VisionGuard</title>

      <div className="page">
        <header className="page-head">
          <h2 className="page-title">Live monitoring</h2>
          <p className="page-intro">
            The annotated camera feed with every detection and track in the current frame.
            Boxes and track IDs are drawn into the video by the backend.
          </p>
          <MagicToggle />
        </header>

        {/* Same grid as the dashboard; DOM order is the mobile reading order. */}
        <div className={`layout${reachable ? '' : ' is-dimmed'}`}>
          <div className="column column-main">
            <LiveStream
              viewState={status.viewState}
              active={visible}
              restartCount={status.restartCount}
              resolution={frame?.resolution}
              sourceLabel={selectedCamera?.name ?? data?.info.source_id}
              camera={cameraParam}
            />
            <FrameResults frame={frame} />
          </div>
          <div className="column column-side">
            <StatusPanel status={status} frame={frame} />
            <StatsPanel stats={stats} eventCount={data?.event_count ?? null} />
            <ModulesPanel
              modules={data?.info.modules ?? []}
              latency={stats?.module_latency_ms}
              errors={frame?.errors}
              live={status.viewState === 'live'}
            />
          </div>
        </div>
      </div>
    </>
  );
}
