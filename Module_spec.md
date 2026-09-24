# VisionGuard AI — Module Specification

## Module Design Principle

Every module must perform one clearly defined task.

A module must not silently perform responsibilities belonging to another module.

---

# Module Categories

## A. Detection Modules

### 1. Person Detection

Purpose:
Detect persons in a frame.

Input:
Frame

Output:
Person detections.

Output fields:

- class
- confidence
- bounding_box
- frame_id
- timestamp

Dependencies:
None.

---

### 2. Object Detection

Purpose:
Detect configured object classes.

Input:
Frame

Output:
Object detections.

Dependencies:
None.

---

### 3. PPE Detection

Purpose:
Detect PPE items.

Possible classes:

- helmet
- safety vest
- gloves
- safety shoes

Input:
Frame

Output:
PPE detections.

Dependencies:
None at module level.

---

### 4. Fire Detection

Purpose:
Detect visual fire candidates.

Input:
Frame

Output:
Fire detections/confidence.

---

### 5. Smoke Detection

Purpose:
Detect visual smoke candidates.

Input:
Frame

Output:
Smoke detections/confidence.

---

### 6. Vehicle Detection

Purpose:
Detect configured vehicles.

Possible classes:

- car
- truck
- bus
- forklift
- motorcycle

---

### 7. Crowd Detection

Purpose:
Provide people-count/crowd-related visual analysis.

May consume:
Person Detection

---

# B. Recognition Modules

## 8. Face Detection

Purpose:

Locate faces.

Input:
Frame

Output:
Face bounding boxes.

Does NOT identify people.

---

## 9. Face Recognition

Purpose:

Match detected faces with registered identities.

Input:

Face crop / face embedding

Output:

Identity match + similarity/confidence.

Does NOT perform attendance.

---

## 10. Unknown Person

Purpose:

Determine whether a detected/recognized face matches an allowed/registered identity.

Consumes:

Face Detection
Face Recognition

Produces:

Known / Unknown state.

---

# C. Tracking Modules

## 11. Person Tracking

Purpose:

Maintain persistent person track IDs across frames.

Input:

Person detections.

Output:

Tracked persons.

---

## 12. Object Tracking

Purpose:

Maintain persistent object track IDs.

Input:

Object detections.

---

## 13. Vehicle Tracking

Purpose:

Maintain persistent vehicle track IDs.

Input:

Vehicle detections.

---

# D. Higher-Level Features

These are not standalone low-level detectors.

## Attendance

Consumes:

- Face Detection
- Face Recognition
- Tracking
- Time

Produces:

- check-in
- check-out
- presence state

---

## PPE Violation

Consumes:

- Person Detection
- PPE Detection
- Tracking
- optional zone information

Produces:

PPE violation event.

---

## Unknown Person Alert

Consumes:

- Person Detection
- Face Detection
- Face Recognition
- optional zone information

Produces:

Unknown-person event.

---

## Restricted Zone

Consumes:

- Person/Object/Vehicle Tracking
- configured polygon zone

Produces:

Zone entry/exit/dwell events.

---

# Module Independence Rule

A low-level AI module must not directly:

- send notifications
- modify attendance
- create business incidents
- access dashboard state
- contain unrelated business rules

It should return structured results.

Higher-level systems consume those results.