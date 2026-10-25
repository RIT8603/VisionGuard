/** Browser helpers for face enrolment: camera availability and frame capture. */

export type CameraSupport = 'ok' | 'insecure' | 'unsupported';

/**
 * Browsers expose the camera only on a secure origin (localhost or HTTPS).
 * Over plain HTTP on a network address - e.g. a Tailscale or LAN IP - the
 * page has to use photo uploads instead.
 */
export function cameraSupport(): CameraSupport {
  if (!window.isSecureContext) return 'insecure';
  if (!navigator.mediaDevices?.getUserMedia) return 'unsupported';
  return 'ok';
}

/** The backend shrinks larger images to this anyway (faces.MAX_SIDE). */
const MAX_SIDE = 1280;

/** The current video frame as a JPEG, unmirrored (the preview is mirrored by CSS only). */
export function captureFrame(video: HTMLVideoElement, quality = 0.9): Promise<Blob | null> {
  const { videoWidth: width, videoHeight: height } = video;
  if (!width || !height) return Promise.resolve(null);
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const context = canvas.getContext('2d');
  if (!context) return Promise.resolve(null);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
}

/** A friendly message for a getUserMedia failure. */
export function cameraErrorMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : '';
  if (name === 'NotAllowedError') return 'Camera permission was denied. Allow it in the browser, or use Upload photos.';
  if (name === 'NotFoundError') return 'No camera was found. Use Upload photos instead.';
  if (name === 'NotReadableError') return 'The camera is in use by another application.';
  return 'The camera could not be started. Use Upload photos instead.';
}
