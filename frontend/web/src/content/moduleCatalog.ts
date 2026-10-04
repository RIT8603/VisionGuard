/**
 * Customer-facing catalogue of VisionGuard capabilities.
 *
 * Each entry maps a business feature to the backend module keys it needs
 * (the names under `modules:` in configs/config.yaml). Sources of truth:
 *   available / setup  repository README "AI modules" table + configs/config.yaml
 *   planned (roadmap)  docs/ROADMAP.md, README "Still to build", Agent.md §5
 *   planned (vision)   the owner's product architecture diagram only
 * Update this file when a module's status changes in the backend.
 */

export type ModuleStatus = 'available' | 'setup' | 'planned';

/** Where a planned item comes from, so the page can say how firm it is. */
export type PlanSource = 'roadmap' | 'vision';

export type ModuleCategory = 'Detection & tracking' | 'Safety & analysis' | 'Identity';

export interface CatalogModule {
  id: string;
  name: string;
  category: ModuleCategory;
  status: ModuleStatus;
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
  available: 'Available',
  setup: 'Requires setup',
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
    summary: 'Finds people in every frame and follows each one with a persistent ID.',
    backendModules: ['person_detection', 'person_tracking'],
  },
  {
    id: 'objects',
    name: 'Object tracking',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Detects and tracks the 80 everyday object classes of the COCO dataset.',
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
    note: 'Counts only; crowd events are planned.',
    backendModules: ['person_detection', 'crowd_detection'],
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
    summary: 'Counts chosen item types in view and raises an event when a count leaves its allowed range.',
    note: 'Items and limits are set in the configuration file. Counts what the camera can see, so it suits shelves and bays viewed face-on.',
    backendModules: ['object_detection', 'inventory_monitoring'],
  },
  {
    id: 'low-light',
    name: 'Low-light / night detection',
    category: 'Detection & tracking',
    status: 'available',
    summary: 'Raises an event when the scene goes dark and when it recovers.',
    note: 'Can also brighten dark frames before detection, but that is off by default: in testing it did not improve detection.',
    backendModules: ['low_light'],
  },

  // ---------- safety & analysis ----------
  {
    id: 'zones',
    name: 'Restricted zones',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Raises events when a tracked person enters, leaves or lingers in an area you define.',
    note: 'Zones are drawn as polygons in the configuration file.',
    backendModules: ['person_detection', 'person_tracking', 'restricted_zone'],
  },
  {
    id: 'falls',
    name: 'Fall detection',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Estimates body pose and raises an event when someone appears to have fallen.',
    note: 'A heuristic for a person to verify, not a medical device.',
    backendModules: ['pose_detection', 'fall_detection'],
  },
  {
    id: 'ppe',
    name: 'PPE detection',
    category: 'Safety & analysis',
    status: 'setup',
    summary: 'Detects helmets, vests and other protective equipment.',
    note: 'Needs a custom-trained model; no stock model has these classes. Violation events are planned.',
    backendModules: ['ppe_detection'],
  },
  {
    id: 'fire-smoke',
    name: 'Fire & smoke detection',
    category: 'Safety & analysis',
    status: 'setup',
    summary: 'Detects visible fire and smoke in the scene.',
    note: 'Needs custom-trained models; the project does not ship them.',
    backendModules: ['fire_detection', 'smoke_detection'],
  },
  {
    id: 'illegal-activity',
    name: 'Illegal activity: after-hours intrusion',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Raises a critical event when a person is present during hours nobody should be.',
    note: 'Decided from place and time only. It does not recognise actions such as fighting or theft - that needs a model this project does not have.',
    backendModules: ['person_detection', 'person_tracking', 'intrusion_detection'],
  },
  {
    id: 'loitering',
    name: 'Loitering / dwell-time alert',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Raises an event when a person stays in roughly one place longer than a set time.',
    backendModules: ['person_detection', 'person_tracking', 'loitering'],
  },
  {
    id: 'tailgating',
    name: 'Tailgating / piggybacking detection',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Raises an event when two people cross an entry line back to back, and counts people in and out.',
    note: 'The entry line is drawn in the configuration file. It cannot see badges, so it flags a pair for review rather than proving a breach.',
    backendModules: ['person_detection', 'person_tracking', 'tailgating'],
  },
  {
    id: 'abandoned-object',
    name: 'Abandoned object detection',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Raises an event when a bag or suitcase sits still with nobody near it.',
    note: 'A heuristic: an owner sitting a few metres away looks the same as a bag left behind.',
    backendModules: ['person_detection', 'object_detection', 'object_tracking', 'abandoned_object'],
  },
  {
    id: 'weapons',
    name: 'Weapon detection',
    category: 'Safety & analysis',
    status: 'available',
    summary: 'Detects knives, bats and scissors and raises a critical event.',
    note: 'The stock model has no gun class - firearms need a custom-trained model. Expect misses and false alarms on small objects.',
    backendModules: ['weapon_detection'],
  },

  // ---------- identity ----------
  {
    id: 'faces',
    name: 'Face recognition',
    category: 'Identity',
    status: 'setup',
    summary: 'Identifies enrolled people by face. Stores face embeddings, never images.',
    note: 'Needs two face models (OpenCV Zoo) and an identity store built on purpose. Off by default.',
    backendModules: ['face_detection', 'face_recognition'],
  },
  {
    id: 'attendance',
    name: 'Automated attendance',
    category: 'Identity',
    status: 'setup',
    summary: 'Turns recognised faces into check-in and check-out events.',
    note: 'Builds on face recognition, so it needs the same models and identity store.',
    backendModules: ['face_detection', 'face_recognition', 'attendance'],
  },
  {
    id: 'unknown-person',
    name: 'Unknown-person alerts',
    category: 'Identity',
    status: 'setup',
    summary: 'Raises an event when a face is seen repeatedly that matches no one enrolled.',
    note: 'Builds on face recognition. Events appear in the portal; no notifications are sent.',
    backendModules: ['face_detection', 'face_recognition', 'unknown_person'],
  },
  {
    id: 'reid',
    name: 'Multi-camera re-identification',
    category: 'Identity',
    status: 'setup',
    summary: 'Recognises a person again after they leave and return, or on another camera, from clothing and build.',
    note: 'Needs a 107 MB model you download, and one VisionGuard per camera sharing a gallery file. Not a face match: about two in three correct on test footage.',
    backendModules: ['person_detection', 'person_tracking', 'person_reid'],
  },
];

export const CATEGORIES: readonly ModuleCategory[] = [
  'Detection & tracking',
  'Safety & analysis',
  'Identity',
];
