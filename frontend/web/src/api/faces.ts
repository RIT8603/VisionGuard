/**
 * Face Registry calls (/api/face/*). Separate from endpoints.ts so the
 * registry's contract stays in one place.
 *
 * Images go as the raw request body (a JPEG Blob), the way the backend
 * expects; nothing is written to disk on either side.
 */

import { sendJson } from './client';
import {
  isAccuracyResponse,
  isAddSampleResponse,
  isDeleteUserResponse,
  isFaceReading,
  isFaceStatus,
  isFaceUser,
  isFaceUsersResponse,
  isIdentifyResponse,
  type AccuracyResponse,
  type AddSampleResponse,
  type FacePose,
  type FaceReading,
  type FaceStatus,
  type FaceUser,
  type FaceUserFields,
  type IdentifyResponse,
} from './faceTypes';

const FACE = '/api/face';

/** The first image request loads the face models on the server. */
const IMAGE_TIMEOUT_MS = 15000;

interface Options {
  signal?: AbortSignal;
}

export function getFaceStatus({ signal }: Options = {}): Promise<FaceStatus> {
  return sendJson('GET', `${FACE}/status`, undefined, isFaceStatus, { signal });
}

export async function listFaceUsers({ signal }: Options = {}): Promise<FaceUser[]> {
  return (await sendJson('GET', `${FACE}/users`, undefined, isFaceUsersResponse, { signal })).users;
}

export function getFaceUser(id: number, { signal }: Options = {}): Promise<FaceUser> {
  return sendJson('GET', `${FACE}/users/${id}`, undefined, isFaceUser, { signal });
}

export function createFaceUser(fields: FaceUserFields): Promise<FaceUser> {
  return sendJson('POST', `${FACE}/users`, fields, isFaceUser);
}

export function updateFaceUser(
  id: number,
  changes: Partial<FaceUserFields> & { active?: boolean },
): Promise<FaceUser> {
  return sendJson('PATCH', `${FACE}/users/${id}`, changes, isFaceUser);
}

/** Removes the user and every template of theirs. */
export async function deleteFaceUser(id: number): Promise<void> {
  await sendJson('DELETE', `${FACE}/users/${id}`, undefined, isDeleteUserResponse);
}

/**
 * Learn one template from `image`. `pose` 'auto' files it under whichever view
 * the photo shows (uploads); a named pose is refused if the head is turned
 * another way. A refusal's reason is the ApiError message.
 */
export function addFaceSample(
  id: number,
  pose: FacePose | 'auto',
  image: Blob,
  { signal }: Options = {},
): Promise<AddSampleResponse> {
  const query = new URLSearchParams({ pose });
  return sendJson('POST', `${FACE}/users/${id}/samples?${query}`, image, isAddSampleResponse, {
    signal,
    timeoutMs: IMAGE_TIMEOUT_MS,
  });
}

/** Remove a user's templates for one pose, or all of them when `pose` is omitted. */
export function clearFaceSamples(id: number, pose?: FacePose): Promise<FaceUser> {
  const query = pose ? `?${new URLSearchParams({ pose })}` : '';
  return sendJson('DELETE', `${FACE}/users/${id}/samples${query}`, undefined, isFaceUser);
}

/** Judge an image for enrolment without storing anything (drives the live guide). */
export function analyzeFace(image: Blob, { signal }: Options = {}): Promise<FaceReading> {
  return sendJson('POST', `${FACE}/analyze`, image, isFaceReading, {
    signal,
    timeoutMs: IMAGE_TIMEOUT_MS,
  });
}

/** The live matcher's answer for one photo. Stores nothing. */
export function identifyFace(image: Blob, { signal }: Options = {}): Promise<IdentifyResponse> {
  return sendJson('POST', `${FACE}/identify`, image, isIdentifyResponse, {
    signal,
    timeoutMs: IMAGE_TIMEOUT_MS,
  });
}

export function getFaceAccuracy({ signal }: Options = {}): Promise<AccuracyResponse> {
  return sendJson('GET', `${FACE}/accuracy`, undefined, isAccuracyResponse, {
    signal,
    timeoutMs: IMAGE_TIMEOUT_MS,
  });
}
