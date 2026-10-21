"""HTTP routes for the face registry (``/api/face/*``).

Kept out of ``server.py`` so that file stays about video and monitoring. The
routes are thin: every rule (validation, quality gates, duplicate detection,
rebuilding the live store) lives in :class:`backend.services.faces.FaceService`.

Images arrive as the raw request body (``image/jpeg`` or ``image/png``), the
same way video uploads do - no multipart dependency, and the browser can send a
canvas ``Blob`` directly. Nothing sent is written to disk: an image is decoded,
turned into a numeric template, and dropped.
"""

from __future__ import annotations

from typing import Optional

from backend.persistence.face_db import DuplicateUserError, FaceDbError, UserNotFoundError
from backend.services.faces import FaceService, FaceServiceError, MAX_IMAGE_BYTES
from computer_vision.utils.logger import get_logger

try:
    # Module level, not inside the function: with postponed annotations FastAPI
    # resolves ``Request`` / ``Body`` / ``Query`` hints through this module's globals.
    from fastapi import Body, Query, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - dashboard dependencies missing
    Body = Query = Request = JSONResponse = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


def register_face_routes(app, faces: Optional[FaceService]) -> None:
    """Attach the ``/api/face`` routes to ``app``.

    With ``faces`` unset (tests, a run without a registry) every route answers
    503 instead of vanishing, so the page can say why.
    """
    def unavailable() -> JSONResponse:
        return JSONResponse({"error": "The face registry is not available in this run."}, 503)

    def fail(exc: Exception) -> JSONResponse:
        if isinstance(exc, FaceServiceError):
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        if isinstance(exc, UserNotFoundError):
            return JSONResponse({"error": str(exc)}, status_code=404)
        if isinstance(exc, DuplicateUserError):
            return JSONResponse({"error": str(exc)}, status_code=409)
        if isinstance(exc, FaceDbError):
            return JSONResponse({"error": str(exc)}, status_code=400)
        raise exc

    async def read_image(request: Request) -> bytes | JSONResponse:
        """The request body, refusing early anything larger than an image should be."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_IMAGE_BYTES:
            return JSONResponse({"error": "That image is too large (limit 8 MB)"}, 413)
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                return JSONResponse({"error": "That image is too large (limit 8 MB)"}, 413)
            chunks.append(chunk)
        return b"".join(chunks)

    @app.get("/api/face/status")
    def face_status():
        if faces is None:
            return unavailable()
        return JSONResponse(faces.status())

    @app.get("/api/face/users")
    def face_users():
        if faces is None:
            return unavailable()
        return JSONResponse({"users": faces.list_users()})

    @app.post("/api/face/users", include_in_schema=False)
    def face_create_user(payload: dict = Body(...)):
        if faces is None:
            return unavailable()
        try:
            return JSONResponse(faces.create_user(payload), status_code=201)
        except FaceDbError as exc:
            return fail(exc)

    @app.get("/api/face/users/{user_id}")
    def face_get_user(user_id: int):
        if faces is None:
            return unavailable()
        try:
            return JSONResponse(faces.get_user(user_id))
        except FaceDbError as exc:
            return fail(exc)

    @app.patch("/api/face/users/{user_id}", include_in_schema=False)
    def face_update_user(user_id: int, payload: dict = Body(...)):
        if faces is None:
            return unavailable()
        try:
            return JSONResponse(faces.update_user(user_id, payload))
        except FaceDbError as exc:
            return fail(exc)

    @app.delete("/api/face/users/{user_id}", include_in_schema=False)
    def face_delete_user(user_id: int):
        if faces is None:
            return unavailable()
        try:
            faces.delete_user(user_id)
            return JSONResponse({"deleted": user_id})
        except FaceDbError as exc:
            return fail(exc)

    @app.post("/api/face/analyze", include_in_schema=False)
    async def face_analyze(request: Request):
        """Report on one image (face, pose, quality) without storing anything."""
        if faces is None:
            return unavailable()
        body = await read_image(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            return JSONResponse(faces.analyze(body))
        except FaceDbError as exc:
            return fail(exc)

    @app.post("/api/face/identify", include_in_schema=False)
    async def face_identify(request: Request):
        """Who is in this photo, according to the live matcher? Stores nothing."""
        if faces is None:
            return unavailable()
        body = await read_image(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            return JSONResponse(faces.identify(body))
        except FaceDbError as exc:
            return fail(exc)

    @app.get("/api/face/accuracy")
    def face_accuracy():
        """Leave-one-out accuracy estimate over the registered templates."""
        if faces is None:
            return unavailable()
        return JSONResponse(faces.accuracy())

    @app.post("/api/face/users/{user_id}/samples", include_in_schema=False)
    async def face_add_sample(
        user_id: int, request: Request, pose: str = Query("auto", max_length=16)
    ):
        """Learn one template for a user from the image in the body."""
        if faces is None:
            return unavailable()
        body = await read_image(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            return JSONResponse(faces.add_sample(user_id, pose, body), status_code=201)
        except FaceDbError as exc:
            return fail(exc)

    @app.delete("/api/face/users/{user_id}/samples", include_in_schema=False)
    def face_clear_samples(user_id: int, pose: Optional[str] = Query(None, max_length=16)):
        """Remove a user's templates for one pose, or all of them."""
        if faces is None:
            return unavailable()
        try:
            return JSONResponse(faces.clear_samples(user_id, pose))
        except FaceDbError as exc:
            return fail(exc)
