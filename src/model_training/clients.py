from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping


class JsonHttpClient:
    def __init__(self, base_url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = timeout

    def request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        data = None
        headers = {"Accept": "application/json", **self.headers}
        if payload is not None:
            data = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        return None if not raw else json.loads(raw.decode("utf-8"))


class LiveServiceClient:
    def __init__(self, base_url: str, operator_token: str, *, timeout: float = 10.0) -> None:
        self.http = JsonHttpClient(
            base_url,
            headers={"Authorization": f"Bearer {operator_token}"},
            timeout=timeout,
        )

    def cameras(self) -> list[dict[str, Any]]:
        payload = self.http.request("GET", "/cameras-status")
        return list(payload.get("cameras", []))

    def capture(self, camera_index: int) -> dict[str, Any]:
        return dict(
            self.http.request(
                "POST",
                f"/operator/api/cameras/{camera_index}/product-training-captures",
            )
        )


class ShopCatalogClient:
    def __init__(self, base_url: str, api_key: str, shop_id: int, *, timeout: float = 10.0) -> None:
        self.shop_id = shop_id
        self.http = JsonHttpClient(
            base_url,
            headers={"X-Api-Key": api_key},
            timeout=timeout,
        )

    def products(self) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"shopId": self.shop_id})
        payload = self.http.request("GET", f"/shop-api/model-training/products?{query}")
        if isinstance(payload, dict):
            return list(payload.get("products", []))
        return list(payload)

