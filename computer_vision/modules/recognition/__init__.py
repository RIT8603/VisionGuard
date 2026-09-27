"""Recognition modules.

"Where is the face?" and "whose face is this?" are deliberately separate
modules. Planned contents (see ROADMAP.md, Phase 7):

    face_detection.py   -> FaceDetectionModule    (locates faces)
    face_recognition.py -> FaceRecognitionModule  (matches identities)
    unknown_person.py   -> UnknownPersonModule    (consumes the two above)

Empty in Phase 1. Identity processing stays off unless explicitly enabled.
"""
