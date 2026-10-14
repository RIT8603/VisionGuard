import type { CameraInfo, CameraInput, SourceType } from '../api/types';
import { backendKeysFor, catalogIdsFor } from '../content/moduleCatalog';

/**
 * The editable form of one camera.
 *
 * Fields are strings (what an <input> holds) and every source type keeps its
 * own field, so switching a slot from RTSP to webcam and back does not lose
 * what was typed. Only the field the chosen source uses is ever sent.
 */
export interface CameraDraft {
  /** Client-side React key; never sent. */
  key: string;
  /** The backend's id for a camera that already exists. Absent for a new slot. */
  id?: string;
  name: string;
  source: SourceType;
  rtspUrl: string;
  youtubeUrl: string;
  deviceId: string;
  path: string;
  loop: boolean;
  /** False: follow the Modules page. True: run exactly `moduleIds`. */
  ownModules: boolean;
  /** Catalogue ids chosen for this camera; kept while `ownModules` is off. */
  moduleIds: string[];
}

let nextKey = 1;
const newKey = () => `slot-${nextKey++}`;

export function blankDraft(): CameraDraft {
  return {
    key: newKey(),
    name: '',
    source: 'rtsp',
    rtspUrl: '',
    youtubeUrl: '',
    deviceId: '0',
    path: '',
    loop: false,
    ownModules: false,
    moduleIds: [],
  };
}

/** A draft for a camera the backend already runs. The URL arrives with its password masked. */
export function draftFromCamera(camera: CameraInfo): CameraDraft {
  return {
    key: newKey(),
    id: camera.id,
    // The backend falls back to the id when there is no name; do not turn that
    // fallback into a stored name on the next apply.
    name: camera.name === camera.id ? '' : camera.name,
    source: camera.source,
    rtspUrl: camera.rtsp_url ?? '',
    youtubeUrl: camera.youtube_url ?? '',
    deviceId: String(camera.device_id ?? 0),
    path: camera.path ?? '',
    loop: camera.loop ?? false,
    ownModules: Array.isArray(camera.modules),
    moduleIds: Array.isArray(camera.modules) ? catalogIdsFor(camera.modules) : [],
  };
}

/** The request body entry for a draft: only the field its source type uses. */
export function toInput(draft: CameraDraft): CameraInput {
  const input: CameraInput = { source: draft.source };
  if (draft.id) input.id = draft.id;
  const name = draft.name.trim();
  if (name) input.name = name;

  if (draft.source === 'rtsp') {
    input.rtsp_url = draft.rtspUrl.trim();
  } else if (draft.source === 'youtube') {
    input.youtube_url = draft.youtubeUrl.trim();
  } else if (draft.source === 'webcam') {
    input.device_id = Number.parseInt(draft.deviceId, 10);
  } else {
    input.path = draft.path.trim();
    if (draft.loop) input.loop = true;
  }
  // Dependencies included; an empty list means "video only" for this camera.
  if (draft.ownModules) input.modules = backendKeysFor(new Set(draft.moduleIds));
  return input;
}

const RTSP_PREFIX = /^rtsps?:\/\//i;
const YOUTUBE_REGEX =
  /^https?:\/\/(www\.|m\.)?(youtube\.com\/(watch\?v=|live\/|embed\/|shorts\/)|youtu\.be\/)[\w-]+/i;

/** What is wrong with every draft, by index; an entry is null when the draft is fine. */
export function validateDrafts(drafts: readonly CameraDraft[]): (string | null)[] {
  const devices = new Map<number, number>();
  return drafts.map((draft, index) => {
    if (draft.source === 'rtsp') {
      const url = draft.rtspUrl.trim();
      if (!url) return 'Enter the camera address.';
      if (!RTSP_PREFIX.test(url)) return 'The address must start with rtsp:// or rtsps://';
      return null;
    }
    if (draft.source === 'youtube') {
      const url = draft.youtubeUrl.trim();
      if (!url) return 'Enter the YouTube stream URL.';
      if (!YOUTUBE_REGEX.test(url)) {
        return 'Enter a valid YouTube stream URL (e.g. https://www.youtube.com/watch?v=... or https://www.youtube.com/live/...).';
      }
      return null;
    }
    if (draft.source === 'file') {
      return draft.path.trim() ? null : 'Enter the path of the video file.';
    }
    const device = Number(draft.deviceId);
    if (draft.deviceId.trim() === '' || !Number.isInteger(device) || device < 0) {
      return 'The device number must be 0 or higher.';
    }
    const firstUse = devices.get(device);
    if (firstUse !== undefined) return `Webcam ${device} is already used by camera ${firstUse + 1}.`;
    devices.set(device, index);
    return null;
  });
}

/** A stable string for comparing "what is on screen" with "what was last applied". */
export function signature(drafts: readonly CameraDraft[]): string {
  return JSON.stringify(drafts.map(toInput));
}
