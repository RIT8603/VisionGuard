import { PATHS } from '../../app/paths';
import { useCameraRegistry } from '../../hooks/useCameraRegistry';
import { Panel } from '../ui/Panel';
import { PanelLink } from '../ui/PanelLink';

interface CamerasSummaryPanelProps {
  /** The single pipeline's source id from /api/status, if known. */
  sourceId: string | undefined;
}

/**
 * Dashboard entry point to the camera registry. Shows counts only when the
 * backend provides a registry; until then it says so instead of guessing.
 */
export function CamerasSummaryPanel({ sourceId }: CamerasSummaryPanelProps) {
  const registry = useCameraRegistry();

  return (
    <>
      <Panel
        title="Cameras"
        aside={registry.state === 'ready' ? `${registry.cameras.length} registered` : 'Awaiting backend'}
      >
        {registry.state === 'ready' ? (
          <p className="camera-summary-text">
            The backend reports {registry.cameras.length} camera
            {registry.cameras.length === 1 ? '' : 's'}.
          </p>
        ) : (
          <p className="camera-summary-text">
            Camera registry data is awaiting backend support. The backend runs a single pipeline
            today
            {sourceId !== undefined && (
              <>
                , source <span className="mono">{sourceId}</span>
              </>
            )}
            .
          </p>
        )}
      </Panel>
      <PanelLink to={PATHS.portal.cameras}>Cameras →</PanelLink>
    </>
  );
}
