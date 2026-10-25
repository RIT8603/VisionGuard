/**
 * Face Registry (/api/face/*) response shapes and their runtime checks.
 *
 * Mirrors backend/services/faces.py and backend/persistence/face_db.py by
 * hand, like types.ts does for the rest of the API. Update both together.
 * The checks are shallow, as in guards.ts: they cover what the UI branches on.
 */

/** The five enrolment views, in the order the backend walks them. */
export type FacePose = 'front' | 'left' | 'right' | 'up' | 'down';

export interface FacePoseInfo {
  id: FacePose;
  /** Instruction for the person, e.g. "Turn your head to your left". */
  label: string;
  required: boolean;
}

export interface FaceModelFile {
  path: string;
  present: boolean;
}

export interface FaceStatus {
  /** false = dashboard.allow_face_edit is off: no writes. */
  editable: boolean;
  models: {
    detection: FaceModelFile;
    recognition: FaceModelFile;
    ready: boolean;
    /** Why loading failed, once a request has tried. */
    error: string | null;
  };
  poses: FacePoseInfo[];
  samples_per_pose: number;
  max_samples_per_pose: number;
  threshold: number;
  users: number;
  enrolled: number;
  templates: number;
}

/** Editable text fields and the backend's length limits (face_db.USER_FIELDS). */
export const FACE_USER_FIELDS = {
  name: 80,
  employee_id: 40,
  role: 60,
  department: 60,
  notes: 500,
} as const;

export type FaceUserFields = { [K in keyof typeof FACE_USER_FIELDS]: string };

export interface FaceUser extends FaceUserFields {
  id: number;
  /** Inactive users are kept but never recognised. */
  active: boolean;
  created_at: string;
  updated_at: string;
  /** Templates stored per pose. */
  poses: Partial<Record<FacePose, number>>;
  sample_count: number;
  /** Every required pose has at least one template. */
  enrolled: boolean;
  missing_poses: FacePose[];
}

/** The main face in an image. `box` is [x1, y1, x2, y2] as fractions of the image. */
export interface FaceBox {
  box: [number, number, number, number];
  confidence: number;
  /** Detected view, or "other" between two views. */
  pose: FacePose | 'other' | null;
  yaw: number;
  pitch: number;
  size_px: number;
  sharpness: number;
  brightness: number;
  quality: number;
}

/** POST /api/face/analyze: what is in an image, judged for enrolment. */
export interface FaceReading {
  image: { width: number; height: number };
  faces: number;
  /** Fit to enrol from: one clear face, no problems. */
  ok: boolean;
  /** Reasons a capture would be refused, worded for the person. */
  problems: string[];
  face: FaceBox | null;
}

export interface AddSampleResponse {
  pose: FacePose;
  quality: number;
  user: FaceUser;
}

export interface IdentifyResponse {
  /** A registered name, or "unknown". */
  decision: string;
  reason: string;
  threshold: number;
  matches: { name: string; similarity: number }[];
  face: FaceBox | null;
  quality_problems: string[];
}

export interface SimilarityStats {
  min: number;
  mean: number;
  max: number;
}

export interface AccuracyResponse {
  identities: number;
  trials: number;
  rank1_accuracy: number | null;
  false_reject_rate: number | null;
  false_accept_rate: number | null;
  genuine: SimilarityStats | null;
  impostor: SimilarityStats | null;
  threshold: number;
  suggested_threshold: number | null;
  note: string;
}

// ---------- runtime checks ----------

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function isFaceStatus(value: unknown): value is FaceStatus {
  return (
    isObject(value) &&
    typeof value.editable === 'boolean' &&
    isObject(value.models) &&
    typeof value.models.ready === 'boolean' &&
    Array.isArray(value.poses) &&
    value.poses.every((pose) => isObject(pose) && typeof pose.id === 'string') &&
    typeof value.samples_per_pose === 'number' &&
    typeof value.threshold === 'number'
  );
}

export function isFaceUser(value: unknown): value is FaceUser {
  return (
    isObject(value) &&
    typeof value.id === 'number' &&
    typeof value.name === 'string' &&
    typeof value.active === 'boolean' &&
    isObject(value.poses) &&
    typeof value.enrolled === 'boolean' &&
    Array.isArray(value.missing_poses)
  );
}

export function isFaceUsersResponse(value: unknown): value is { users: FaceUser[] } {
  return isObject(value) && Array.isArray(value.users) && value.users.every(isFaceUser);
}

export function isDeleteUserResponse(value: unknown): value is { deleted: number } {
  return isObject(value) && typeof value.deleted === 'number';
}

export function isFaceReading(value: unknown): value is FaceReading {
  return (
    isObject(value) &&
    typeof value.ok === 'boolean' &&
    Array.isArray(value.problems) &&
    (value.face === null || (isObject(value.face) && Array.isArray(value.face.box)))
  );
}

export function isAddSampleResponse(value: unknown): value is AddSampleResponse {
  return isObject(value) && typeof value.pose === 'string' && isFaceUser(value.user);
}

export function isIdentifyResponse(value: unknown): value is IdentifyResponse {
  return (
    isObject(value) &&
    typeof value.decision === 'string' &&
    typeof value.reason === 'string' &&
    Array.isArray(value.matches)
  );
}

export function isAccuracyResponse(value: unknown): value is AccuracyResponse {
  return (
    isObject(value) &&
    typeof value.identities === 'number' &&
    typeof value.trials === 'number' &&
    typeof value.threshold === 'number'
  );
}
