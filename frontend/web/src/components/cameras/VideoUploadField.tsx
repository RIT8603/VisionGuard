import { useEffect, useRef, useState } from 'react';
import { getVideos, uploadVideo } from '../../api/endpoints';
import type { UploadedVideo } from '../../api/types';

interface VideoUploadFieldProps {
  id: string;
  disabled: boolean;
  /** The current path, to mark which uploaded video is in use. */
  path: string;
  onPick: (path: string) => void;
}

const formatSize = (bytes: number) =>
  bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(1)} GB`
    : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;

/**
 * Send a video from this browser to the machine running VisionGuard, or pick
 * one sent earlier. Either way the camera's path is filled in for the
 * operator, so they never have to know where the server keeps the file.
 */
export function VideoUploadField({ id, disabled, path, onPick }: VideoUploadFieldProps) {
  const [videos, setVideos] = useState<UploadedVideo[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getVideos(controller.signal)
      .then((body) => setVideos(body.videos))
      .catch(() => {
        /* the list is a convenience; uploading still works */
      });
    return () => {
      controller.abort();
      abortRef.current?.abort();
    };
  }, []);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setProgress(0);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const saved = await uploadVideo(file, setProgress, controller.signal);
      setVideos((current) => [saved, ...current.filter((v) => v.path !== saved.path)]);
      onPick(saved.path);
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
    } finally {
      setProgress(null);
      abortRef.current = null;
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const busy = progress !== null;

  return (
    <div className="field">
      <label htmlFor={`${id}-upload`}>Upload a video</label>
      <input
        ref={inputRef}
        id={`${id}-upload`}
        type="file"
        accept="video/*,.mp4,.avi,.mkv,.mov,.m4v,.webm,.mpg,.mpeg,.ts"
        disabled={disabled || busy}
        onChange={(event) => void onFile(event.target.files?.[0])}
      />
      {busy && (
        <p className="field-help" role="status">
          Uploading… {Math.round((progress ?? 0) * 100)}%{' '}
          <button type="button" className="link-button" onClick={() => abortRef.current?.abort()}>
            Cancel
          </button>
        </p>
      )}
      {error && (
        <p className="field-problem" role="alert">
          {error}
        </p>
      )}
      {videos.length > 0 && (
        <>
          <label htmlFor={`${id}-uploaded`}>Or use one already uploaded</label>
          <select
            id={`${id}-uploaded`}
            value={videos.some((v) => v.path === path) ? path : ''}
            disabled={disabled || busy}
            onChange={(event) => event.target.value && onPick(event.target.value)}
          >
            <option value="">Choose a video…</option>
            {videos.map((video) => (
              <option key={video.path} value={video.path}>
                {video.name} ({formatSize(video.size)})
              </option>
            ))}
          </select>
        </>
      )}
      <p className="field-help">
        The file is saved on the machine running VisionGuard, then Apply starts it.
      </p>
    </div>
  );
}
