import type { Detection, FrameSummary, Track } from '../api/types';
import { fmtBox, fmtPercent } from '../lib/format';
import { Panel } from './Panel';

/** A crowd should not turn the panel into a wall of rows. */
const MAX_ROWS = 12;

// Cells carry data-label so narrow screens can stack each row as a labelled
// block (see .data-table in styles.css) instead of scrolling sideways.

function MoreRows({ total }: { total: number }) {
  return total > MAX_ROWS ? <p className="more-rows mono">+{total - MAX_ROWS} more</p> : null;
}

function GroupHeader({ module, count, noun }: { module: string; count: number; noun: string }) {
  return (
    <h3 className="group-head">
      <span className="mono">{module}</span>
      <span className="group-count">
        {count} {noun}
        {count === 1 ? '' : 's'}
      </span>
    </h3>
  );
}

function TrackTable({ module, tracks }: { module: string; tracks: Track[] }) {
  return (
    <div className="result-group">
      <GroupHeader module={module} count={tracks.length} noun="track" />
      {tracks.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th className="num">ID</th>
              <th>Class</th>
              <th className="num">Conf</th>
              <th className="num">Age</th>
              <th className="num">Hits</th>
              <th>Box x1,y1 → x2,y2</th>
            </tr>
          </thead>
          <tbody>
            {tracks.slice(0, MAX_ROWS).map((track) => (
              <tr key={track.track_id}>
                <td className="num mono cell-id" data-label="ID">#{track.track_id}</td>
                <td data-label="Class">{track.class}</td>
                <td className="num mono" data-label="Conf">{fmtPercent(track.confidence)}</td>
                <td className="num mono" data-label="Age">{track.age}</td>
                <td className="num mono" data-label="Hits">{track.hits}</td>
                <td className="mono" data-label="Box">{fmtBox(track.bbox)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <MoreRows total={tracks.length} />
    </div>
  );
}

function identityOf(detection: Detection): string {
  const identity = detection.extra?.identity;
  return typeof identity === 'string' ? identity : 'unknown';
}

function DetectionTable({ module, detections }: { module: string; detections: Detection[] }) {
  // face_recognition attaches an identity; other detectors do not.
  const showIdentity = detections.some((d) => d.extra !== undefined && 'identity' in d.extra);

  return (
    <div className="result-group">
      <GroupHeader module={module} count={detections.length} noun="detection" />
      {detections.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Class</th>
              {showIdentity && <th>Identity</th>}
              <th className="num">Conf</th>
              <th>Box x1,y1 → x2,y2</th>
            </tr>
          </thead>
          <tbody>
            {detections.slice(0, MAX_ROWS).map((detection, index) => (
              // Detections have no id; order within one frame is stable enough.
              <tr key={index}>
                <td data-label="Class">{detection.class}</td>
                {showIdentity && <td data-label="Identity">{identityOf(detection)}</td>}
                <td className="num mono" data-label="Conf">{fmtPercent(detection.confidence)}</td>
                <td className="mono" data-label="Box">{fmtBox(detection.bbox)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <MoreRows total={detections.length} />
    </div>
  );
}

export function FrameResults({ frame }: { frame: FrameSummary | null }) {
  if (!frame) {
    return (
      <Panel title="Current frame">
        <p className="empty">No frame yet.</p>
      </Panel>
    );
  }

  const trackGroups = Object.entries(frame.tracks);
  const detectionGroups = Object.entries(frame.detections);
  const poseModules = Object.keys(frame.poses);
  const poseCount = Object.values(frame.poses).reduce((sum, list) => sum + list.length, 0);

  return (
    <Panel
      title="Current frame"
      aside={
        <span className="mono">
          F{frame.frame_id} · {frame.resolution.width}×{frame.resolution.height}
        </span>
      }
    >
      {trackGroups.length === 0 && detectionGroups.length === 0 && (
        <p className="empty">No detection or tracking module produced results.</p>
      )}
      {trackGroups.map(([module, tracks]) => (
        <TrackTable key={module} module={module} tracks={tracks} />
      ))}
      {detectionGroups.map(([module, detections]) => (
        <DetectionTable key={module} module={module} detections={detections} />
      ))}
      {poseModules.length > 0 && (
        <p className="more-rows mono">
          {poseCount} pose{poseCount === 1 ? '' : 's'} from {poseModules.join(', ')}
        </p>
      )}
    </Panel>
  );
}
