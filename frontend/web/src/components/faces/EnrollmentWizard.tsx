import { useCallback, useEffect, useRef, useState } from 'react';
import { toApiError } from '../../api/client';
import { addFaceSample, analyzeFace, clearFaceSamples } from '../../api/endpoints';
import type { FacePoseInfo, FaceReading, FaceStatus, FaceUser } from '../../api/types';

/** Longest side of a frame sent for analysis: plenty for a face, small to send. */
const FRAME_MAX_WIDTH = 960;
const JPEG_QUALITY = 0.88;
/** Consecutive good readings of the right pose before a capture is taken. */
const STABLE_READINGS = 2;
/** Minimum gap between captures, so the two shots of a pose differ a little. */
const CAPTURE_GAP_MS = 1500;
const ANALYSE_GAP_MS = 250;

type CameraState = 'off' | 'starting' | 'on' | 'error';
type Mode = 'camera' | 'upload';

interface EnrollmentWizardProps {
  user: FaceUser;
  status: FaceStatus;
  /** Called with the user's new state after every capture or clear. */
  onUserChange: (user: FaceUser) => void;
  onClose: () => void;
}

interface UploadResult {
  name: string;
  ok: boolean;
  text: string;
}

const POSE_SHORT: Record<string, string> = {
  front: 'Front',
  left: 'Left',
  right: 'Right',
  up: 'Up',
  down: 'Down',
};

/**
 * Captures a user's face from several angles.
 *
 * Two ways in: the live camera, which watches the head pose and takes each
 * shot by itself once the person is facing the requested way and the picture is
 * good; or uploaded photos, each filed under whichever view it shows. Either
 * way the server makes the final call - it re-checks pose and quality and
 * refuses a face that already belongs to someone else - and only a numeric
 * template is kept, never the image.
 */
export function EnrollmentWizard({ user, status, onUserChange, onClose }: EnrollmentWizardProps) {
  const [mode, setMode] = useState<Mode>('camera');
  const [camera, setCamera] = useState<CameraState>('off');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [reading, setReading] = useState<FaceReading | null>(null);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null);
  const [paused, setPaused] = useState(false);
  const [skipped, setSkipped] = useState<ReadonlySet<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [uploads, setUploads] = useState<UploadResult[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const perPose = status.samples_per_pose;
  const countOf = (pose: string) => user.poses[pose] ?? 0;
  const target: FacePoseInfo | null =
    status.poses.find((pose) => countOf(pose.id) < perPose && !skipped.has(pose.id)) ?? null;
  const requiredDone = status.poses.filter((p) => p.required).every((p) => countOf(p.id) > 0);
  // Every view still short of captures. The instruction asks for the first,
  // but whichever of them the person happens to show is taken - people rarely
  // turn exactly in the order asked, and waiting for it wastes the good frames.
  const needed = status.poses
    .filter((pose) => countOf(pose.id) < perPose && !skipped.has(pose.id))
    .map((pose) => pose.id);

  // The capture loop reads these through refs, so it never works from a stale
  // render and does not have to restart on every change.
  const targetRef = useRef(target);
  const neededRef = useRef(needed);
  const userRef = useRef(user);
  const pausedRef = useRef(paused);
  targetRef.current = target;
  neededRef.current = needed;
  userRef.current = user;
  pausedRef.current = paused;

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamera('off');
    setReading(null);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamera('error');
      setCameraError(
        'This browser cannot open the camera here. Camera capture needs https or localhost - use "Upload photos" instead.',
      );
      return;
    }
    setCamera('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play().catch(() => undefined);
      }
      setCamera('on');
    } catch (caught) {
      setCamera('error');
      const name = caught instanceof DOMException ? caught.name : '';
      setCameraError(
        name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow it in the browser, or use "Upload photos".'
          : name === 'NotFoundError'
            ? 'No camera was found on this device. Use "Upload photos" instead.'
            : 'The camera could not be opened. It may be in use by another program.',
      );
    }
  }, []);

  // Release the camera when leaving this view or switching to upload.
  useEffect(() => stopCamera, [stopCamera]);
  useEffect(() => {
    if (mode === 'upload') stopCamera();
  }, [mode, stopCamera]);

  const grabFrame = useCallback((): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return Promise.resolve(null);
    const scale = Math.min(1, FRAME_MAX_WIDTH / video.videoWidth);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext('2d');
    if (!context) return Promise.resolve(null);
    // Not mirrored: the server's idea of left and right is the camera's.
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
  }, []);

  const capture = useCallback(
    async (blob: Blob, pose: string) => {
      setBusy(true);
      try {
        const result = await addFaceSample(userRef.current.id, pose, blob);
        onUserChange(result.user);
        setNotice({
          kind: 'ok',
          text: `Captured ${POSE_SHORT[result.pose] ?? result.pose} view (quality ${Math.round(result.quality * 100)}%).`,
        });
        return true;
      } catch (caught) {
        setNotice({ kind: 'error', text: toApiError(caught).message });
        return false;
      } finally {
        setBusy(false);
      }
    },
    [onUserChange],
  );

  // The live loop: look at the camera a few times a second, and take each shot
  // the moment the person is in the requested pose and the picture is good.
  useEffect(() => {
    if (camera !== 'on') return;
    let cancelled = false;
    let timer: number | undefined;
    let stable = 0;
    let stablePose: string | null = null;
    let lastCapture = 0;
    const controller = new AbortController();

    const tick = async () => {
      try {
        const blob = await grabFrame();
        if (blob && !cancelled) {
          const next = await analyzeFace(blob, controller.signal);
          if (cancelled) return;
          setReading(next);

          // Any view still needed counts, not only the one being asked for.
          const pose = next.ok ? next.face?.pose ?? null : null;
          const useful = pose !== null && neededRef.current.includes(pose);
          stable = useful && pose === stablePose ? stable + 1 : useful ? 1 : 0;
          stablePose = useful ? pose : null;
          const ready =
            useful && stable >= STABLE_READINGS && Date.now() - lastCapture >= CAPTURE_GAP_MS;
          if (ready && pose && !pausedRef.current) {
            stable = 0;
            lastCapture = Date.now();
            const saved = await capture(blob, pose);
            // A refusal (another person's face, a duplicate) needs the
            // operator, not a retry loop hammering the server.
            if (!saved) setPaused(true);
          }
        }
      } catch (caught) {
        if (cancelled || controller.signal.aborted) return;
        setReading(null);
        setNotice({ kind: 'error', text: toApiError(caught).message });
      }
      if (!cancelled) timer = window.setTimeout(tick, ANALYSE_GAP_MS);
    };

    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [camera, grabFrame, capture]);

  const captureNow = async () => {
    if (!target) return;
    const blob = await grabFrame();
    if (!blob) {
      setNotice({ kind: 'error', text: 'The camera has not produced a picture yet.' });
      return;
    }
    setPaused(false);
    await capture(blob, target.id);
  };

  const onFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    setUploads([]);
    const results: UploadResult[] = [];
    for (const file of Array.from(files)) {
      try {
        const result = await addFaceSample(user.id, 'auto', file);
        onUserChange(result.user);
        results.push({
          name: file.name,
          ok: true,
          text: `Added as ${POSE_SHORT[result.pose] ?? result.pose} view`,
        });
      } catch (caught) {
        results.push({ name: file.name, ok: false, text: toApiError(caught).message });
      }
      setUploads([...results]);
    }
    setBusy(false);
  };

  const clearPose = async (pose?: string) => {
    setBusy(true);
    try {
      onUserChange(await clearFaceSamples(user.id, pose));
      setNotice({ kind: 'ok', text: pose ? `Cleared the ${pose} captures.` : 'Cleared all captures.' });
      setPaused(false);
    } catch (caught) {
      setNotice({ kind: 'error', text: toApiError(caught).message });
    } finally {
      setBusy(false);
    }
  };

  const face = reading?.face ?? null;
  // Green whenever the face shows a view that is still needed (it will be taken).
  const facing = face?.pose != null && needed.includes(face.pose);
  const box = face?.box;

  return (
    <section className="panel enroll" aria-label={`Face enrolment for ${user.name}`}>
      <header className="panel-head">
        <h2>Face enrolment · {user.name}</h2>
        <span className="panel-aside">
          <button type="button" className="summary-btn summary-btn-quiet enroll-close" onClick={onClose}>
            Done
          </button>
        </span>
      </header>

      <div className="panel-body enroll-body">
        <div className="enroll-main">
          <div className="enroll-modes" role="group" aria-label="Capture method">
            {(['camera', 'upload'] as const).map((option) => (
              <button
                key={option}
                type="button"
                className={`seg-btn${mode === option ? ' is-active' : ''}`}
                aria-pressed={mode === option}
                onClick={() => setMode(option)}
              >
                {option === 'camera' ? 'Use camera' : 'Upload photos'}
              </button>
            ))}
          </div>

          {mode === 'camera' ? (
            <>
              <div className="enroll-stage">
                {/* Mirrored like a mirror, so turning left looks like turning left. */}
                <video ref={videoRef} className="enroll-video" muted playsInline />
                {camera === 'on' && box && (
                  <span
                    className={`enroll-box${face && reading?.ok && facing ? ' is-good' : ''}`}
                    style={{
                      left: `${(1 - box[2]) * 100}%`,
                      top: `${box[1] * 100}%`,
                      width: `${(box[2] - box[0]) * 100}%`,
                      height: `${(box[3] - box[1]) * 100}%`,
                    }}
                  />
                )}
                {camera !== 'on' && (
                  <div className="enroll-cover">
                    {camera === 'starting' ? (
                      <p>Opening the camera…</p>
                    ) : (
                      <>
                        <p>{cameraError ?? 'The camera is off.'}</p>
                        <button type="button" className="summary-btn" onClick={() => void startCamera()}>
                          {camera === 'error' ? 'Try again' : 'Start camera'}
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>

              {camera === 'on' && (
                <div className="enroll-guide" aria-live="polite">
                  {target ? (
                    <p className="enroll-instruction">
                      <strong>{target.label}.</strong>{' '}
                      {facing ? 'Hold still…' : 'Move until the box turns green.'}
                    </p>
                  ) : (
                    <p className="enroll-instruction">
                      <strong>All views captured.</strong> Press Done, or add more with Upload photos.
                    </p>
                  )}
                  <ul className="enroll-problems">
                    {(reading?.problems ?? []).slice(0, 2).map((problem) => (
                      <li key={problem}>{problem}</li>
                    ))}
                    {reading && reading.problems.length === 0 && target && !facing && face && (
                      <li>
                        Looks like the {face.pose ?? 'unknown'} view — {target.label.toLowerCase()}.
                      </li>
                    )}
                  </ul>
                  <div className="enroll-actions">
                    <button
                      type="button"
                      className="summary-btn"
                      disabled={busy || !target || !reading?.ok || !status.editable}
                      onClick={() => void captureNow()}
                    >
                      Capture now
                    </button>
                    {paused && (
                      <button
                        type="button"
                        className="summary-btn summary-btn-quiet enroll-resume"
                        onClick={() => setPaused(false)}
                      >
                        Resume auto-capture
                      </button>
                    )}
                    <button
                      type="button"
                      className="summary-btn summary-btn-quiet enroll-resume"
                      onClick={stopCamera}
                    >
                      Stop camera
                    </button>
                  </div>
                  {paused && (
                    <p className="field-help">Auto-capture is paused until you resume it.</p>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="enroll-upload">
              <p className="field-help">
                Choose clear photos of {user.name}: one person, face fully visible, and ideally a mix of
                front, left, right, up and down views. Each photo is filed under the view it shows.
              </p>
              <input
                type="file"
                accept="image/jpeg,image/png"
                multiple
                disabled={busy || !status.editable}
                onChange={(event) => {
                  void onFiles(event.target.files);
                  event.target.value = '';
                }}
              />
              {uploads.length > 0 && (
                <ul className="enroll-results">
                  {uploads.map((result, index) => (
                    <li key={`${result.name}-${index}`} className={result.ok ? 'is-ok' : 'is-bad'}>
                      <span className="mono">{result.name}</span> — {result.text}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {notice && (
            <p className={notice.kind === 'ok' ? 'enroll-note' : 'inline-error'} role="status">
              {notice.text}
            </p>
          )}
        </div>

        <aside className="enroll-side">
          <h3 className="enroll-side-title">Views</h3>
          <ol className="pose-list">
            {status.poses.map((pose) => {
              const have = countOf(pose.id);
              const isTarget = mode === 'camera' && camera === 'on' && target?.id === pose.id;
              const done = have >= perPose;
              return (
                <li
                  key={pose.id}
                  className={`pose-row${isTarget ? ' is-target' : ''}${done ? ' is-done' : ''}`}
                >
                  <span className="pose-name">
                    {POSE_SHORT[pose.id] ?? pose.id}
                    {!pose.required && <span className="pose-optional"> optional</span>}
                  </span>
                  <span className="pose-dots mono" aria-label={`${have} of ${perPose} captured`}>
                    {Array.from({ length: perPose }, (_, index) => (index < have ? '●' : '○')).join(' ')}
                  </span>
                  <span className="pose-tools">
                    {!pose.required && !done && have === 0 && (
                      <button
                        type="button"
                        className="link-btn"
                        onClick={() =>
                          setSkipped((previous) => {
                            const next = new Set(previous);
                            if (next.has(pose.id)) next.delete(pose.id);
                            else next.add(pose.id);
                            return next;
                          })
                        }
                      >
                        {skipped.has(pose.id) ? 'Unskip' : 'Skip'}
                      </button>
                    )}
                    {have > 0 && status.editable && (
                      <button
                        type="button"
                        className="link-btn"
                        disabled={busy}
                        onClick={() => void clearPose(pose.id)}
                      >
                        Retake
                      </button>
                    )}
                  </span>
                </li>
              );
            })}
          </ol>

          <p className={requiredDone ? 'enroll-status is-ok' : 'enroll-status'}>
            {requiredDone
              ? 'Enrolled — front, left and right are captured.'
              : 'Not enrolled yet — front, left and right are required.'}
          </p>
          {user.sample_count > 0 && status.editable && (
            <button
              type="button"
              className="link-btn link-btn-danger"
              disabled={busy}
              onClick={() => void clearPose()}
            >
              Clear all captures
            </button>
          )}
          <p className="field-help">
            Only a numeric face template is stored — never the photo. Better coverage across angles and
            lighting means more reliable recognition.
          </p>
        </aside>
      </div>
    </section>
  );
}
