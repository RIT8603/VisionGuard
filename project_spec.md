# VisionGuard AI — Project Specification

## 1. Overview

VisionGuard AI is a configurable real-time computer vision platform.

The platform accepts video input from cameras and analyzes the video using selected AI modules.

The same platform can be configured for different requirements by enabling different AI modules.

---

# 2. Core Concept

The platform consists of:

Camera Source
→ Vision Core
→ AI Modules
→ Analysis
→ Dashboard / Alerts

The user does not need every AI capability.

The user selects the required modules.

---

# 3. Camera Sources

Supported sources:

### Webcam

Used for local development and testing.

### Video File

Used for testing, debugging and evaluation.

### RTSP

Used for CCTV/IP camera deployment.

Architecture:

BaseVideoSource
├── WebcamSource
├── VideoFileSource
└── RTSPSource

---

# 4. Vision Core

The Vision Core is responsible for:

- capturing frames
- preprocessing
- frame management
- FPS measurement
- latency measurement
- annotation
- AI pipeline orchestration

OpenCV is the primary computer-vision framework.

---

# 5. AI Modules

## Detection

### Person Detection

Detect humans.

### Object Detection

Detect configured objects.

### PPE Detection

Detect configured protective equipment.

### Fire Detection

Detect fire.

### Smoke Detection

Detect smoke.

### Vehicle Detection

Detect vehicles.

### Crowd Detection

Detect/count people for crowd analysis.

---

## Recognition

### Face Detection

Locate faces.

### Face Recognition

Match detected faces against registered identities.

### Unknown Person

Identify faces that cannot be matched to configured known identities.

---

## Tracking

### Person Tracking

Maintain persistent IDs for people.

### Object Tracking

Maintain persistent IDs for objects.

### Vehicle Tracking

Maintain persistent IDs for vehicles.

---

# 6. Higher-Level Features

Some features are not basic AI detection modules.

Examples:

Attendance
Intrusion
PPE Violation
Restricted Zone Alert
Unknown Person Alert
Crowd Alert

These features consume outputs from one or more AI modules.

---

# 7. Example

Attendance may use:

Face Detection
+
Face Recognition
+
Person Tracking
+
Time

PPE Violation may use:

Person Detection
+
PPE Detection
+
Tracking
+
Zone Analysis
+
Temporal Rules

Unknown Person Alert may use:

Person Detection
+
Face Detection
+
Face Recognition
+
Zone
+
Rules

---

# 8. Module Selection

Each camera/configuration can specify enabled modules.

Example:

modules:
  person_detection: true
  person_tracking: true
  ppe_detection: true
  fire_detection: true
  face_detection: false
  face_recognition: false

Only enabled modules should execute.

---

# 9. Real-Time Output

The system should provide:

- detection results
- tracking results
- confidence
- FPS
- latency
- current active modules
- events when higher-level logic is enabled

The dashboard will consume structured results.

---

# 10. Current MVP

Current implementation:

Webcam
→ OpenCV
→ Person Detection
→ Person Tracking
→ Live Annotated Output

Future modules will be added incrementally.

---

# 11. Future Production Flow

RTSP Camera
→ VisionGuard Capture
→ OpenCV Vision Core
→ Selected AI Modules
→ Analysis
→ Event Intelligence
→ Dashboard
→ Alerts
→ Evidence / Incident Storage