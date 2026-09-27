# VisionGuard AI — Development Roadmap

## Phase 1 — Foundation

- project structure
- configuration
- logging
- BaseVideoSource
- WebcamSource
- FrameContext
- BaseAIModule
- ModuleRegistry

Status:
CURRENT

---

## Phase 2 — Person Detection

Implement:

- PersonDetectionModule
- model loading
- inference
- bounding boxes
- confidence
- FPS

---

## Phase 3 — Person Tracking

Implement:

- PersonTrackingModule
- persistent track IDs
- tracking visualization
- tracking performance

---

## Phase 4 — Video File Source

Implement:

- VideoFileSource
- configuration
- seek/loop support if required

---

## Phase 5 — RTSP Source

Implement:

- RTSPSource
- reconnect
- timeout
- dropped-frame handling

---

## Phase 6 — Additional Detection

Implement incrementally:

- Object Detection
- PPE Detection
- Fire Detection
- Smoke Detection
- Vehicle Detection
- Crowd Detection

---

## Phase 7 — Recognition

Implement:

- Face Detection
- Face Recognition
- Unknown Person

---

## Phase 8 — Additional Tracking

Implement:

- Object Tracking
- Vehicle Tracking

---

## Phase 9 — Higher-Level Features

Implement:

- Attendance
- Restricted Zone
- PPE Violation
- Unknown Person Alert
- Crowd Events

---

## Phase 10 — Dashboard

Implement:

- camera management
- module selection
- live video
- module results
- events
- alerts
- configuration

---

## Phase 11 — Cloud

Later:

- AWS
- S3
- DynamoDB
- API Gateway
- Lambda
- SNS
- CloudWatch

---

## Phase 12 — Agentic Vision

Only after the deterministic vision pipeline is stable.

---

# Scope Rule

Never jump directly to a later phase unless explicitly requested.

Current priority:

Phase 1 → Phase 2 → Phase 3