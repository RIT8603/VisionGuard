import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { toApiError } from '../../api/client';
import type { FacePose, FaceReading, FaceStatus, FaceUser } from '../../api/faceTypes';
import { addFaceSample, analyzeFace, clearFaceSamples } from '../../api/faces';
import { cameraErrorMessage, cameraSupport, captureFrame } from '../../lib/faceCapture';
import { Panel } from '../ui/Panel';

/** Delay between live-guide frames; one request is in flight at a time. */
const ANALYZE_EVERY_MS = 600;
/** Pause after a capture or a refusal, so the person can move on. */
const AFTER_CAPTURE_MS = 1200;
const AFTER_REFUSAL_MS = 2000;
/** The backend's limit (faces.MAX_IMAGE_BYTES). */
const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;

type Message = { tone: 'ok' | 'warn' | 'crit'; text: string };
type UploadResult = { file: string; ok: boolean; text: string };

interface Props {
  user: FaceUser;
  status: FaceStatus;
  onUserChange: (user: FaceUser) => void;
}

/**
 * Multi-angle enrolment for one user. The camera watches the head pose and
 * captures each needed view itself once the server says the picture is good;
 * uploads are filed under whichever view they show. Order does not matter.
 */
export function EnrollmentWizard({ user, status, onUserChange }: Props) {
  const [cameraOn, setCameraOn] = useState(false);
  const [reading, setReading] = useState<FaceReading | null>(null);
  const [message, setMessage] = useState<Message | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploads, setUploads] = useState<UploadResult[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  // The capture loop reads these, so it never acts on a stale user.
  const userRef = useRef(user);
  userRef.current = user;

  const support = cameraSupport();
  const canWrite = status.editable && status.models.ready;
  const perPose = status.samples_per_pose;
  const required = status.poses.filter((pose) => pose.required);
  const count = (pose: FacePose) => user.poses[pose] ?? 0;

  /** Views still worth capturing: required ones first, then the optional ones. */
  const neededPoses = (current: FaceUser): FacePose[] => {
    const short = (ids: FacePose[]) => ids.filter((id) => (current.poses[id] ?? 0) < perPose);
    const missingRequired = short(required.map((pose) => pose.id));
    if (missingRequired.length) return missingRequired;
    return short(status.poses.filter((pose) => !pose.required).map((pose) => pose.id));
  };
  const nextPose = status.poses.find((pose) => pose.id === neededPoses(user)[0]);

  // ---- camera: open, guide, auto-capture ----
  useEffect(() => {
    if (!cameraOn) return;
    let stream: MediaStream | null = null;
    let timer = 0;
    let stopped = false;
    let pauseUntil = 0;
    const controller = new AbortController();

    const tick = async () => {
      const video = videoRef.current;
      if (stopped || !video) return;
      try {
        const frame = await captureFrame(video);
        if (frame && !stopped) {
          const result = await analyzeFace(frame, { signal: controller.signal });
          if (stopped) return;
          setReading(result);
          const pose = result.face?.pose;
          const wanted = neededPoses(userRef.current);
          if (result.ok && pose && pose !== 'other' && wanted.includes(pose) && Date.now() >= pauseUntil) {
            try {
              const saved = await addFaceSample(userRef.current.id, pose, frame, { signal: controller.signal });
              if (stopped) return;
              const justEnrolled = saved.user.enrolled && !userRef.current.enrolled;
              onUserChange(saved.user);
              setMessage({
                tone: 'ok',
                text: justEnrolled
                  ? 'Enrolled - front, left and right are captured. Continue for the optional up/down views, or press Done.'
                  : `Captured ${saved.pose} (quality ${saved.quality.toFixed(2)})`,
              });
              pauseUntil = Date.now() + AFTER_CAPTURE_MS;
              if (neededPoses(saved.user).length === 0) {
                setCameraOn(false);
                setMessage({ tone: 'ok', text: 'All views captured.' });
                return;
              }
            } catch (caught) {
              if (controller.signal.aborted) return;
              setMessage({ tone: 'crit', text: toApiError(caught).message });
              pauseUntil = Date.now() + AFTER_REFUSAL_MS;
            }
          }
        }
      } catch (caught) {
        if (controller.signal.aborted) return;
        const error = toApiError(caught);
        setMessage({ tone: 'crit', text: error.message });
        // Models missing or registry gone: stop rather than retry forever.
        if (error.status === 503 || error.status === 403) {
          setCameraOn(false);
          return;
        }
      }
      if (!stopped) timer = window.setTimeout(tick, ANALYZE_EVERY_MS);
    };

    navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false })
      .then((media) => {
        if (stopped) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }
        stream = media;
        const video = videoRef.current;
        if (video) {
          video.srcObject = media;
          void video.play().catch(() => undefined);
        }
        timer = window.setTimeout(tick, ANALYZE_EVERY_MS);
      })
      .catch((caught: unknown) => {
        if (stopped) return;
        setMessage({ tone: 'crit', text: cameraErrorMessage(caught) });
        setCameraOn(false);
      });

    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
      stream?.getTracks().forEach((track) => track.stop());
      setReading(null);
    };
    // Restarts only when the camera is switched on or off; the user is read via userRef.
  }, [cameraOn]);

  // ---- uploads ----
  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (!files.length) return;
    setBusy(true);
    setMessage(null);
    const results: UploadResult[] = [];
    for (const file of files) {
      if (file.size > MAX_UPLOAD_BYTES) {
        results.push({ file: file.name, ok: false, text: 'too large (limit 8 MB)' });
        continue;
      }
      try {
        const saved = await addFaceSample(userRef.current.id, 'auto', file);
        onUserChange(saved.user);
        results.push({ file: file.name, ok: true, text: `${saved.pose} (quality ${saved.quality.toFixed(2)})` });
      } catch (caught) {
        results.push({ file: file.name, ok: false, text: toApiError(caught).message });
      }
      setUploads([...results]);
    }
    setUploads(results);
    setBusy(false);
  };

  const clear = async (pose?: FacePose) => {
    const what = pose ? `the ${pose} captures` : `every capture of ${user.name}`;
    if (!window.confirm(`Remove ${what}?`)) return;
    setBusy(true);
    try {
      onUserChange(await clearFaceSamples(user.id, pose));
      setMessage(null);
    } catch (caught) {
      setMessage({ tone: 'crit', text: toApiError(caught).message });
    } finally {
      setBusy(false);
    }
  };

  const box = reading?.face?.box;
  const boxTone = reading?.ok ? 'face-box-ok' : 'face-box-warn';

  return (
    <Panel
      title="Face enrolment"
      aside={user.enrolled ? 'Enrolled' : `Needs ${user.missing_poses.join(', ') || '—'}`}
      className="face-enrol"
    >
      {user.enrolled && (
        <p className="face-done">Enrolled - front, left and right are captured. Up and down are optional.</p>
      )}

      <ol className="face-poses">
        {status.poses.map((pose) => {
          const n = count(pose.id);
          return (
            <li key={pose.id} className={n >= perPose ? 'face-pose-done' : n > 0 ? 'face-pose-part' : undefined}>
              <span className="face-pose-name">
                {pose.id}
                {!pose.required && <span className="muted"> (optional)</span>}
              </span>
              <span className="face-pose-label muted">{pose.label}</span>
              <span className="mono">
                {n}/{perPose}
              </span>
              {canWrite && n > 0 && (
                <button type="button" className="face-link" disabled={busy || cameraOn} onClick={() => void clear(pose.id)}>
                  Retake
                </button>
              )}
            </li>
          );
        })}
      </ol>

      {!status.editable && <p className="page-meta">Read-only: enrolment is turned off for this run.</p>}
      {status.editable && !status.models.ready && (
        <p className="page-meta">Enrolment needs the face models - see Registry above.</p>
      )}

      {canWrite && (
        <>
          {cameraOn && (
            <div className="face-camera">
              <div className="face-preview">
                <video ref={videoRef} autoPlay playsInline muted />
                {box && (
                  <span
                    className={`face-box ${boxTone}`}
                    style={{
                      left: `${box[0] * 100}%`,
                      top: `${box[1] * 100}%`,
                      width: `${(box[2] - box[0]) * 100}%`,
                      height: `${(box[3] - box[1]) * 100}%`,
                    }}
                  />
                )}
              </div>
              <p className="face-guide" aria-live="polite">
                {nextPose ? <strong>{nextPose.label}</strong> : 'All views captured.'}
                {reading && !reading.ok && reading.problems[0] && <span> - {reading.problems[0]}</span>}
                {reading?.ok && reading.face?.pose && <span className="muted"> (seen: {reading.face.pose})</span>}
              </p>
            </div>
          )}

          {message && (
            <p className={`face-message face-message-${message.tone}`} role={message.tone === 'crit' ? 'alert' : 'status'}>
              {message.text}
            </p>
          )}

          <div className="face-actions">
            {support === 'ok' && (
              <button
                type="button"
                className={cameraOn ? 'summary-btn summary-btn-quiet' : 'summary-btn'}
                disabled={busy}
                onClick={() => {
                  setMessage(null);
                  setCameraOn(!cameraOn);
                }}
              >
                {cameraOn ? (user.enrolled ? 'Done' : 'Stop camera') : 'Start camera'}
              </button>
            )}
            <label className={`summary-btn summary-btn-quiet face-upload${busy || cameraOn ? ' is-disabled' : ''}`}>
              {busy ? 'Working…' : 'Upload photos'}
              <input
                type="file"
                accept="image/jpeg,image/png"
                multiple
                disabled={busy || cameraOn}
                onChange={(event) => void upload(event)}
              />
            </label>
            {user.sample_count > 0 && (
              <button
                type="button"
                className="summary-btn summary-btn-quiet face-danger"
                disabled={busy || cameraOn}
                onClick={() => void clear()}
              >
                Clear all
              </button>
            )}
          </div>

          {support !== 'ok' && (
            <p className="page-meta">
              {support === 'insecure'
                ? 'The browser allows the camera only on localhost or HTTPS. On this address, use Upload photos.'
                : 'This browser has no camera access. Use Upload photos.'}
            </p>
          )}
          <p className="field-help">
            Good captures: light from the front, one person in frame, no cap or mask; glasses are fine. Uploads are
            filed under the view each photo shows.
          </p>

          {uploads.length > 0 && (
            <ul className="face-uploads">
              {uploads.map((result, index) => (
                <li key={index} className={result.ok ? 'face-ok' : 'face-bad'}>
                  <span className="mono">{result.file}</span>: {result.ok ? `✓ ${result.text}` : `refused - ${result.text}`}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Panel>
  );
}
