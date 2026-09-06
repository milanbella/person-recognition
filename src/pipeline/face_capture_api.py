from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import Response

from pipeline.face_capture import FaceCaptureCoordinator, FaceCaptureError


def create_face_capture_router(
    *,
    coordinator: FaceCaptureCoordinator,
    api_token: str,
    operator_api_token: str | None = None,
    shop_id: int | None = None,
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

    def translate_error(error: FaceCaptureError) -> HTTPException:
        status = {
            "IDEMPOTENCY_CONFLICT": 409,
            "CAPTURE_ALREADY_ACTIVE": 409,
            "TARGET_MISMATCH": 409,
            "TARGET_AMBIGUOUS": 409,
            "TOO_MANY_CAPTURES": 429,
            "IMAGE_NOT_READY": 409,
            "CAPTURE_EXPIRED": 410,
        }.get(error.code, 422)
        return HTTPException(
            status_code=status,
            detail={"code": error.code, "message": str(error)},
        )

    @router.post("/face-captures", status_code=202)
    def create_capture(
        response: Response,
        payload: dict[str, Any] = Body(...),
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        require_auth(authorization)
        response.headers["Cache-Control"] = "no-store"
        try:
            requested_shop_id = payload.get("shopId")
            if (
                shop_id is not None
                and requested_shop_id is not None
                and int(requested_shop_id) != shop_id
            ):
                raise FaceCaptureError(
                    "SHOP_MISMATCH",
                    "shopId does not match this person-recognition service.",
                )
            visit_value = payload.get("visitId")
            return coordinator.create(
                customer_id=(
                    None if payload.get("customerId") is None else str(payload["customerId"])
                ),
                visit_id=None if visit_value is None else int(visit_value),
                purpose=str(payload.get("purpose", "identity_verification")),
                timeout_seconds=(
                    None
                    if payload.get("timeoutSeconds") is None
                    else float(payload["timeoutSeconds"])
                ),
                idempotency_key=idempotency_key,
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, FaceCaptureError):
                raise translate_error(error) from error
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_REQUEST", "message": str(error)},
            ) from error

    @router.get("/face-captures/{capture_id}")
    def capture_status(
        capture_id: str,
        response: Response,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_auth(authorization)
        response.headers["Cache-Control"] = "no-store"
        try:
            return coordinator.get(capture_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "CAPTURE_NOT_FOUND", "message": "Unknown face capture."},
            ) from error

    @router.get("/face-captures/{capture_id}/image")
    def capture_image(
        capture_id: str,
        authorization: str | None = Header(default=None),
    ) -> Response:
        require_auth(authorization)
        try:
            image, digest = coordinator.image(capture_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "CAPTURE_NOT_FOUND", "message": "Unknown face capture."},
            ) from error
        except FaceCaptureError as error:
            raise translate_error(error) from error
        return Response(
            content=image,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Face-Capture-Id": capture_id,
                "X-Face-Capture-Sha256": digest,
            },
        )

    @router.delete("/face-captures/{capture_id}")
    def delete_capture(
        capture_id: str,
        response: Response,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_auth(authorization)
        response.headers["Cache-Control"] = "no-store"
        return coordinator.delete(capture_id)

    return router
