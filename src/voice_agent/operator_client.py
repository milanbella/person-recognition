from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


class OperatorApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class OperatorApiClient:
    base_url: str
    api_token: str
    timeout_seconds: float = 5.0

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/operator/api/state")

    def voice_context(self, run_id: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(run_id, safe="")
        return self._request(
            "GET",
            f"/operator/api/test-runs/{quoted}/voice-context",
        )

    def observation(self, camera_index: int) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/observer-cameras/{camera_index}/observations",
        )

    def world_state(self) -> dict[str, Any]:
        return self._request("GET", "/world-state")

    def subject_world_state(self, run_id: str, subject_id: str) -> dict[str, Any]:
        quoted_run = urllib.parse.quote(run_id, safe="")
        quoted_subject = urllib.parse.quote(subject_id, safe="")
        return self._request(
            "GET",
            f"/operator/api/test-runs/{quoted_run}/subjects/{quoted_subject}/world-state?captureQuery=true",
        )

    def visit_world_state(self, visit_id: int) -> dict[str, Any]:
        return self._request("GET", f"/world-state/visits/{visit_id}")

    def shelf_world_state(self, shelf_id: int) -> dict[str, Any]:
        return self._request("GET", f"/world-state/shelves/{shelf_id}")

    def product_world_state(self, visit_id: int) -> dict[str, Any]:
        return self._request("GET", f"/world-state/visits/{visit_id}/products")

    def camera_world_state(self, camera_index: int) -> dict[str, Any]:
        return self._request("GET", f"/world-state/cameras/{camera_index}")

    def create_annotation(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        quoted = urllib.parse.quote(run_id, safe="")
        return self._request(
            "POST",
            f"/operator/api/test-runs/{quoted}/annotations",
            payload,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if payload is not None:
            body = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            detail = f"Operator API returned HTTP {exc.code}."
            try:
                error_payload = json.loads(response_body.decode("utf-8"))
                detail = str(error_payload.get("detail", detail))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            raise OperatorApiError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise OperatorApiError(503, f"Operator API unavailable: {exc.reason}") from exc
        try:
            result = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperatorApiError(502, "Operator API returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise OperatorApiError(502, "Operator API returned a non-object response.")
        return result
