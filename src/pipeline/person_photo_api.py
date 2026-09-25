from __future__ import annotations

import base64
import hmac
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response

from pipeline.person_photo import PersonPhotoCapture, PersonPhotoError


def create_person_photo_router(
    *,
    coordinator: PersonPhotoCapture,
    api_token: str,
    operator_api_token: str | None = None,
) -> APIRouter:
    router = APIRouter()
    accepted_tokens = {api_token}
    if operator_api_token:
        accepted_tokens.add(operator_api_token)

    def require_auth(authorization: str | None) -> None:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail={"code": "UNAUTHORIZED", "message": "A valid bearer token is required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        supplied = authorization.removeprefix("Bearer ")
        if not any(hmac.compare_digest(supplied, token) for token in accepted_tokens):
            raise HTTPException(
                status_code=401,
                detail={"code": "UNAUTHORIZED", "message": "A valid bearer token is required."},
                headers={"WWW-Authenticate": "Bearer"},
            )

    def translate_error(error: PersonPhotoError) -> HTTPException:
        status = {
            "TOO_MANY_CAPTURES": 429,
            "CAMERA_NOT_FOUND": 404,
            "PERSON_NOT_VISIBLE": 404,
            "MULTIPLE_PEOPLE": 409,
            "CAMERA_TIMEOUT": 504,
            "CAPTURE_ENCODING_FAILED": 500,
        }.get(error.code, 422)
        return HTTPException(
            status_code=status,
            detail={"code": error.code, "message": str(error)},
        )

    @router.get("/cameras/{camera_index}/person")
    def capture_person_evidence(
        camera_index: int,
        response: Response,
        track_id: int | None = None,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_auth(authorization)
        response.headers["Cache-Control"] = "no-store"
        try:
            jpeg, evidence = coordinator.capture_person_evidence(camera_index, track_id)
        except PersonPhotoError as error:
            raise translate_error(error) from error
        return {**evidence, "image": {
            "contentType": "image/jpeg",
            "base64": base64.b64encode(jpeg).decode("ascii"),
        }}

    @router.get("/cameras/{camera_index}/person.jpg")
    def capture_person(
        camera_index: int,
        track_id: int | None = None,
        authorization: str | None = Header(default=None),
    ) -> Response:
        require_auth(authorization)
        try:
            jpeg, selected_track, sequence = coordinator.capture_person(camera_index, track_id)
        except PersonPhotoError as error:
            raise translate_error(error) from error
        return Response(content=jpeg, media_type="image/jpeg", headers={
            "Cache-Control": "no-store",
            "X-Camera-Index": str(camera_index),
            "X-Track-Id": str(selected_track),
            "X-Rgb-Sequence-Number": str(sequence),
        })

    return router
