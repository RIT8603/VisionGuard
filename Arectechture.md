# VisionGuard AI — Architecture

## High-Level Architecture

Camera Source
      ↓
Video Source Abstraction
      ↓
OpenCV Vision Core
      ↓
AI Module Registry
      ↓
Enabled AI Modules
      ↓
Shared Analysis Context
      ↓
Higher-Level Analysis
      ↓
Dashboard / Alerts

---

# Camera Source

BaseVideoSource

Implementations:

- WebcamSource
- VideoFileSource
- RTSPSource

All sources return frames through the same interface.

---

# Vision Core

Responsibilities:

- capture
- preprocessing
- timing
- FPS
- frame metadata
- pipeline orchestration

---

# AI Module Registry

The registry discovers and initializes enabled modules.

Example:

PersonDetectionModule
PersonTrackingModule

Disabled modules must not initialize.

---

# Shared Context

FrameContext contains:

- frame
- frame_id
- timestamp
- source_id
- detections
- tracks
- metadata

Modules can read/write their defined results.

---

# Important Dependency Rule

Detection modules should generally run before tracking modules that consume their detections.

Example:

Person Detection
      ↓
Person Tracking

Face Detection
      ↓
Face Recognition

Person Detection
      +
PPE Detection
      +
Tracking
      ↓
Higher-Level PPE Analysis

---

# No Circular Dependencies

Modules must not depend on each other in circular ways.

Example of BAD architecture:

PersonDetection
→ Attendance
→ FaceRecognition
→ PersonDetection

Avoid this.

---

# Future Architecture

The platform may later include:

Event Intelligence
Incident Management
Evidence Storage
Dashboard
Notifications
AWS services
Agentic Vision

These are outside the initial local prototype.