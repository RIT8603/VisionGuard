/**
 * Customer-facing catalogue of VisionGuard capabilities.
 *
 * Each entry maps a business feature to the backend module keys it needs
 * (the names under `modules:` in configs/config.yaml). Sources of truth:
 *   available / setup / experimental  backend implementations, local models and configs/config.yaml
 *   planned (roadmap)  docs/ROADMAP.md, README "Still to build", Agent.md §5
 *   planned (vision)   the owner's product architecture diagram only
 * Update this file when a module's status changes in the backend.
 */

export type ModuleStatus = 'available' | 'setup' | 'experimental' | 'planned';

/** Where a planned item comes from, so the page can say how firm it is. */
export type PlanSource = 'roadmap' | 'vision';

export type ModuleCategory = 'Detection & tracking' | 'Safety & analysis' | 'Identity';

export interface CatalogModule {
  id: string;
  name: string;
  category: ModuleCategory;
  status: ModuleStatus;
  /** Short prerequisite shown beside the camera checkbox; not a live health check. */
  requirement?: string;
  summary: string;
  /** What must happen before it can run (setup), or what exists so far (planned). */
  note?: string;
  planSource?: PlanSource;
  /**
   * Backend module keys to enable, dependencies included, in pipeline order.
   * Empty for planned items: nothing exists to enable yet.
   */
  backendModules: string[];
}

export const STATUS_LABEL: Record<ModuleStatus, string> = {
  available: 'Implemented',
  setup: 'Needs model setup',
  experimental: 'Experimental',
  planned: 'Planned',
};

export const PLAN_SOURCE_LABEL: Record<PlanSource, string> = {
  roadmap: 'On the development roadmap',
  vision: 'Product vision, not yet scheduled',
};

export const MODULE_CATALOG: readonly CatalogModule[] = [
  // ---------- detection & tracking ----------
  {
    id: 'people',
    name: 'People detection & tracking',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Detects visible people and follows them across analysed frames with track IDs.',
    note: 'Blur, occlusion and leaving the view can interrupt detection or change an ID. A track ID is not a registered identity.',
    backendModules: ['person_detection', 'person_tracking'],
  },
  {
    id: 'objects',
    name: 'Object tracking',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Detects and tracks configured everyday object classes from the COCO model.',
    note: 'The current configuration excludes people, vehicles, animals and weapon classes here to avoid duplicate or misleading boxes. Those supported by separate modules use their own detectors.',
    backendModules: ['object_detection', 'object_tracking'],
  },
  {
    id: 'vehicles',
    name: 'Car / vehicle monitoring',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Detects and tracks cars, trucks, buses, motorcycles and bicycles.',
    note: 'Movement only. Number-plate reading is a separate module.',
    backendModules: ['vehicle_detection', 'vehicle_tracking'],
  },
  {
    id: 'crowd',
    name: 'Crowd counting',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Keeps a live count of people in view against a configurable threshold.',
    note: 'Raises an event once a crowd has lasted a few seconds, and again when it clears.',
    backendModules: ['person_detection', 'crowd_detection', 'crowd_alert'],
  },
  {
    id: 'anpr',
    name: 'Number-plate recognition (ANPR)',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Finds the plate on each tracked vehicle and reads it when it is large enough to be legible.',
    note: 'Needs the plate about 80 px wide in the camera image - a gate or entry camera, not a wide street view. Smaller plates are boxed but not read. Plate numbers are personal data.',
    backendModules: ['vehicle_detection', 'vehicle_tracking', 'number_plate_recognition'],
  },
  {
    id: 'inventory',
    name: 'Inventory monitoring',
    category: 'Detection & tracking',
    status: 'available',
    requirement: 'set items / limits',
    summary: 'Counts chosen item types in view and raises an event when a count leaves its allowed range.',
    note: 'Items and limits are set in the configuration file. Counts what the camera can see, so it suits shelves and bays viewed face-on.',
    backendModules: ['object_detection', 'inventory_monitoring'],
  },
  {
    id: 'low-light',
    name: 'Low-light scene monitoring',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Raises an event when the scene goes dark and when it recovers.',
    note: 'Measures scene brightness; this is not infrared night vision. Optional brightening is off by default and cannot recover details missing from the video.',
    backendModules: ['low_light'],
  },

  // ---------- safety & analysis ----------
  {
    id: 'zones',
    name: 'Restricted zones',
    category: 'Safety & analysis',
    status: 'available',
    requirement: 'configure zones',
    summary: 'Raises events when a tracked person enters, leaves or lingers in an area you define.',
    note: 'Zones are drawn as polygons in the configuration file.',
    backendModules: ['person_detection', 'person_tracking', 'restricted_zone'],
  },
  {
    id: 'falls',
    name: 'Fall detection',
    category: 'Safety & analysis',
    status: 'experimental',
    summary: 'Estimates body pose and raises an event when someone appears to have fallen.',
    note: 'Pose-based heuristic. The pose model is present, but falls still need validation on this camera; sitting, bending and occlusion can confuse it.',
    backendModules: ['pose_detection', 'fall_detection'],
  },
  {
    id: 'ppe',
    name: 'PPE detection & violation alerts',
    category: 'Safety & analysis',
    status: 'experimental',
    requirement: 'validate on this camera',
    summary: 'Detects protective equipment and checks tracked people against required-equipment rules.',
    note: 'A helmet/vest model is configured and real video detections and violation alerts were tested. Distant gear is still missed on factory CCTV. Check camera accuracy before use; this configuration does not enable gloves or boots. Alerts use explicit missing-gear predictions.',
    backendModules: ['person_detection', 'person_tracking', 'ppe_detection', 'ppe_violation'],
  },
  {
    id: 'fire-smoke',
    name: 'Fire & smoke detection',
    category: 'Safety & analysis',
    status: 'setup',
    summary: 'Detects visible fire and smoke in the scene.',
    note: 'No fire or smoke model is configured or bundled locally. Persistence-based event code exists, but real detection and alerts cannot be verified until trained models are supplied.',
    backendModules: ['fire_detection', 'smoke_detection'],
  },
  {
    id: 'illegal-activity',
    name: 'After-hours intrusion alerts',
    category: 'Safety & analysis',
    status: 'available',
    requirement: 'set hours / zones',
    summary: 'Raises a critical event when a person is present during hours nobody should be.',
    note: 'Decided from place and time only. It does not recognise actions such as fighting or theft - that needs a model this project does not have.',
    backendModules: ['person_detection', 'person_tracking', 'intrusion_detection'],
  },
  {
    id: 'loitering',
    name: 'Loitering / dwell-time alert',
    category: 'Safety & analysis',
    status: 'available',
    requirement: 'set dwell time',
    summary: 'Raises an event when a person stays in roughly one place longer than a set time.',
    backendModules: ['person_detection', 'person_tracking', 'loitering'],
  },
  {
    id: 'tailgating',
    name: 'Tailgating / piggybacking detection',
    category: 'Safety & analysis',
    status: 'available',
    requirement: 'configure entry line',
    summary: 'Raises an event when two people cross an entry line back to back, and counts people in and out.',
    note: 'The entry line is drawn in the configuration file. It cannot see badges, so it flags a pair for review rather than proving a breach.',
    backendModules: ['person_detection', 'person_tracking', 'tailgating'],
  },
  {
    id: 'gate-counter',
    name: 'Entry / exit gate counter',
    category: 'Safety & analysis',
    status: 'available',
    requirement: 'configure gate line',
    summary: 'Logs every person crossing the gate line as an entry or exit, with their name when recognised, and keeps a live inside count.',
    note: 'The gate line is set in the configuration file and shared by every camera that uses it. Entries and exits are kept in the event log (Reports page).',
    backendModules: ['person_detection', 'person_tracking', 'gate_counter'],
  },
  {
    id: 'abandoned-object',
    name: 'Abandoned object detection',
    category: 'Safety & analysis',
    status: 'experimental',
    summary: 'Raises an event when a bag or suitcase sits still with nobody near it.',
    note: 'A heuristic: an owner sitting a few metres away looks the same as a bag left behind.',
    backendModules: ['person_detection', 'object_detection', 'object_tracking', 'abandoned_object'],
  },
  {
    id: 'weapons',
    name: 'Weapon detection',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Detects configured weapons and marks their tracked holders as armed.',
    note: 'Knife, pistol and rifle models are present in this installation. Rifle detections and armed-person alerts were tested on the evaluation video. Small, distant or hidden weapons may still be missed.',
    backendModules: ['person_detection', 'person_tracking', 'weapon_detection'],
  },

  // ---------- identity ----------
  {
    id: 'faces',
    name: 'Face recognition',
    category: 'Identity',
    status: 'available',
    requirement: 'enroll faces first',
    summary: 'Identifies enrolled people by face. Stores face embeddings, never images.',
    note: 'Face detection and SFace models are present. Register people on the Faces page and enable this module for the camera. Small, blurred or hidden faces cannot reliably establish identity; body tracking can only retain a previously verified name while continuity holds.',
    backendModules: ['face_detection', 'face_recognition'],
  },
  {
    id: 'attendance',
    name: 'Automated attendance',
    category: 'Identity',
    status: 'available',
    requirement: 'enroll faces first',
    summary: 'Turns recognised faces into check-in and check-out events.',
    note: 'Uses enrolled identities and repeated face matches. Opt-in; tune the absence timeout. Losing sight of a face can eventually cause check-out even if the person remains nearby. Attendance rules are regression-tested; a real attendance session still needs validation.',
    backendModules: ['face_detection', 'face_recognition', 'attendance'],
  },
  {
    id: 'face-liveness',
    name: 'Face liveness check',
    category: 'Identity',
    status: 'experimental',
    summary: 'Looks for blink and landmark-motion cues and flags suspected photo spoofs for review.',
    note: 'A basic passive check that flags a held-up still photo for review. It can also flag a real person standing perfectly still, so by default it alerts rather than blocks; replayed videos and masks are not caught.',
    backendModules: ['face_detection', 'face_liveness', 'face_recognition'],
  },
  {
    id: 'unknown-person',
    name: 'Unknown-person alerts',
    category: 'Identity',
    status: 'available',
    requirement: 'enroll faces first',
    summary: 'Raises an alert with a saved snapshot when a face is seen repeatedly that matches no one enrolled.',
    note: 'Enroll known people first; this is an unmatched-face alert, not proof of an intruder. Needs a sufficiently clear face. Snapshots appear in Reports and warning events can create dashboard alerts; external notifications are not connected.',
    backendModules: ['face_detection', 'face_recognition', 'unknown_person'],
  },
  {
    id: 'reid',
    name: 'Multi-camera re-identification',
    category: 'Identity',
    status: 'experimental',
    requirement: 'validate cross-camera matches',
    summary: 'Recognises a person again after they leave and return, or on another camera, from clothing and build.',
    note: 'The appearance model and shared gallery path are configured; concurrent camera writes are regression-tested. A real two-view trial missed the match at the default threshold. Validate matching on your camera pair; clothing similarity is not a verified identity.',
    backendModules: ['person_detection', 'person_tracking', 'person_reid'],
  },
];

export const CATEGORIES: readonly ModuleCategory[] = [
  'Detection & tracking',
  'Safety & analysis',
  'Identity',
];

/** Backend module keys a set of catalogue ids needs, dependencies included,
 * in catalogue order. Planned items contribute nothing - there is no module
 * to enable yet. */
export function backendKeysFor(selected: ReadonlySet<string>): string[] {
  const keys: string[] = [];
  for (const module of MODULE_CATALOG) {
    if (module.status === 'planned' || !selected.has(module.id)) continue;
    for (const key of module.backendModules) {
      if (!keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

/** True when every backend module this catalogue entry needs is in `keys`. */
export function isCoveredBy(module: CatalogModule, keys: ReadonlySet<string>): boolean {
  return module.backendModules.length > 0 && module.backendModules.every((key) => keys.has(key));
}

/** The catalogue ids a set of backend module keys amounts to. */
export function catalogIdsFor(keys: Iterable<string>): string[] {
  const set = new Set(keys);
  return MODULE_CATALOG.filter((module) => isCoveredBy(module, set)).map((module) => module.id);
}
