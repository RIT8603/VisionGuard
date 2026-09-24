# VisionGuard AI — AI Agent Development Instructions

## 1. Project Identity

Project Name: VisionGuard AI

VisionGuard AI is a modular real-time computer vision platform.

The platform receives video from a camera source, processes the stream through an OpenCV-based vision core, runs selected AI modules, performs real-time analysis, and presents results through the dashboard and alert system.

---

# 2. CORE PRODUCT IDEA

The core concept is:

Camera Input
    ↓
Video Capture
    ↓
OpenCV Vision Core
    ↓
Selected AI Modules
    ↓
Real-Time Analysis
    ↓
Dashboard / Alerts

The system must support multiple independent AI modules.

Each AI module performs a specific computer-vision task.

Modules are selected/enabled based on the user's requirements.

Do NOT assume that every module must run simultaneously.

---

# 3. VIDEO INPUT

The system must support multiple video sources through an abstraction.

Supported sources:

1. Webcam
2. Video File
3. RTSP Camera

Development source:

Webcam

Production source:

RTSP CCTV/IP Camera

The AI modules must NOT depend directly on the video source.

For example:

Webcam
and
RTSP

must produce the same internal FrameContext structure.

Do NOT hard-code cv2.VideoCapture(0) throughout the application.

---

# 4. VISION CORE

OpenCV is the central vision-processing layer.

The Vision Core is responsible for:

- video capture
- frame acquisition
- frame preprocessing
- frame resizing
- frame conversion
- FPS measurement
- latency measurement
- frame annotation
- pipeline orchestration

The Vision Core must NOT contain business-specific logic.

---

# 5. AI MODULE ARCHITECTURE

AI functionality must be modular.

Initial module categories:

## Detection Modules

- Person Detection
- Object Detection
- PPE Detection
- Fire Detection
- Smoke Detection
- Vehicle Detection
- Crowd Detection

## Recognition Modules

- Face Detection
- Face Recognition
- Unknown Person Detection

## Tracking Modules

- Person Tracking
- Object Tracking
- Vehicle Tracking

## Future Analysis Modules

- Pose Detection
- Fall Detection
- Loitering Detection
- Intrusion Analysis

These modules are independent components.

Do not merge unrelated modules into one large class.

---

# 6. CURRENT IMPLEMENTATION SCOPE

IMPORTANT:

Do NOT implement every listed module immediately.

For the first prototype, implement only:

1. Person Detection
2. Person Tracking

Prepare clean interfaces for future modules.

Do NOT implement these yet:

- Face Recognition
- Attendance
- Unknown Person
- PPE
- Fire
- Smoke
- Crowd
- Vehicle
- Alerts
- AWS
- Agentic Vision
- Advanced dashboard

These will be implemented in later phases.

---

# 7. MODULE INDEPENDENCE

Every AI module must have:

- unique module name
- module type
- version
- configuration
- initialization method
- processing method
- output schema
- cleanup method

Example:

BaseAIModule

    initialize()
    process()
    get_result()
    cleanup()

Do not allow one module to directly control another module.

Modules communicate through shared context/results.

---

# 8. MODULE ENABLE/DISABLE

Modules must be configuration-driven.

Example:

modules:
  person_detection:
    enabled: true

  person_tracking:
    enabled: true

  face_detection:
    enabled: false

  face_recognition:
    enabled: false

The application must load only enabled modules.

Disabled modules must NOT initialize their models.

---

# 9. SHARED CONTEXT

Modules should communicate using a shared FrameContext / AnalysisContext.

Example:

FrameContext:

- frame
- frame_id
- timestamp
- source_id
- detections
- tracks
- metadata

Person Detection writes detections.

Person Tracking reads detections and writes tracks.

Future modules can consume existing results.

Do NOT duplicate model inference unnecessarily.

---

# 10. PERSON DETECTION

Person Detection has exactly one primary responsibility:

Detect persons in the current frame.

Output:

- bounding box
- confidence
- class
- timestamp
- frame ID

It must NOT:

- perform attendance
- identify a person
- send alerts
- access the database
- perform business rules

---

# 11. PERSON TRACKING

Person Tracking has exactly one primary responsibility:

Maintain a persistent track ID for detected persons across frames.

Example:

Frame 1:
Person → Track 1

Frame 2:
Person → Track 1

Frame 3:
Person → Track 1

Tracking must be independent from detection.

Do not combine PersonDetection and PersonTracking into a single module.

---

# 12. FACE DETECTION VS FACE RECOGNITION

Keep these separate.

Face Detection:

"Where is the face?"

Face Recognition:

"Whose face is this?"

Do not combine them into one module.

---

# 13. UNKNOWN PERSON

Unknown Person Detection depends on:

Face Detection
+
Face Recognition
+
Identity Database

Do not implement Unknown Person Detection as a separate visual detector.

It is a higher-level feature built from lower-level capabilities.

---

# 14. ATTENDANCE

Attendance is NOT a basic detection module.

It is a higher-level feature that can consume:

Face Detection
Face Recognition
Person Tracking
Time
Location

Do not implement attendance inside Face Recognition.

---

# 15. PPE

PPE Detection is an independent AI module.

It may detect:

- helmet
- safety vest
- gloves
- safety shoes
- other configured PPE

Do not place PPE business rules inside the detector.

Detection should produce structured results.

Higher-level logic can determine whether the detected PPE state represents a violation.

---

# 16. EVENT / BUSINESS LOGIC

AI detection and business logic must remain separate.

Example:

Person Detection:

Person detected.

PPE Detection:

Helmet missing.

Zone Analysis:

Person is inside restricted zone.

Business/Event logic can then determine:

PPE violation inside restricted zone.

Do NOT put this complete business rule inside the PPE detector.

---

# 17. REAL-TIME PROCESSING

The application is intended for real-time video analysis.

The pipeline must measure:

- FPS
- processing latency
- inference latency where possible
- frame count

Do not introduce unnecessary blocking operations inside the frame-processing loop.

---

# 18. VIDEO SOURCE ABSTRACTION

Create:

BaseVideoSource

Implement:

WebcamSource
VideoFileSource
RTSPSource

The rest of the application should interact with BaseVideoSource.

Do not write source-specific code inside AI modules.

---

# 19. ERROR HANDLING

The system must handle:

- camera unavailable
- webcam unavailable
- RTSP connection failure
- frame read failure
- model loading failure
- invalid configuration
- disabled module
- module processing failure

A failure in one optional module should not automatically crash the entire application.

---

# 20. DO NOT OVERENGINEER

Do not introduce:

- microservices
- Kubernetes
- Kafka
- Redis
- complex distributed systems
- unnecessary cloud infrastructure

during the initial local prototype.

Start with a clean modular monolith.

---

# 21. DO NOT CHANGE THE PRODUCT IDEA

The AI agent must NOT independently change the fundamental architecture.

Do not:

- replace OpenCV without approval
- remove modular AI architecture
- hard-code all modules into one pipeline
- make RTSP mandatory during development
- make cloud processing mandatory
- implement unrelated features
- add features simply because they are common in surveillance systems

If a requested feature is not specified, ask before implementing it.

---

# 22. DEVELOPMENT METHOD

Work phase-by-phase.

Never generate the entire project blindly.

For each phase:

1. Explain the goal.
2. Identify files to create/change.
3. Implement only the requested phase.
4. Run tests.
5. Verify functionality.
6. Report what changed.
7. Wait for approval before starting the next major phase.

---

# 23. CODE QUALITY

Use:

- Python type hints
- Pydantic where appropriate
- clear interfaces
- small modules
- dependency injection where useful
- structured logging
- configuration-driven behavior

Avoid:

- giant files
- giant classes
- global state
- duplicated logic
- hard-coded configuration
- model-specific logic in the core

---

# 24. SECURITY / PRIVACY

Do not assume face recognition or identity analysis is enabled.

These are optional modules.

Camera-level module configuration must determine whether identity-related processing is enabled.

Do not store face data unless the corresponding feature explicitly requires it.

---

# 25. SOURCE OF TRUTH

The following documents are authoritative:

1. PROJECT_SPEC.md
2. ARCHITECTURE.md
3. MODULE_SPEC.md
4. ROADMAP.md

If code conflicts with these documents, report the conflict instead of silently changing the architecture.

If requirements are ambiguous, ask for clarification.

Do not invent product requirements.

---

# 26. CURRENT GOAL

Current goal:

Build the VisionGuard AI modular vision engine using the webcam as the initial input.

First working pipeline:

Webcam
  ↓
OpenCV
  ↓
Person Detection
  ↓
Person Tracking
  ↓
Annotated Live Video
  ↓
FPS / Latency

Nothing beyond this scope should be implemented without approval.