from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse

from model_training.capture import CaptureRegistrar
from model_training.clients import LiveServiceClient, ShopCatalogClient
from model_training.exporter import YoloDatasetExporter
from model_training.store import ModelTrainingStore


class ModelTrainingService:
    def __init__(
        self,
        *,
        state_root: Path,
        api_token: str,
        shop_id: int,
        live_client: LiveServiceClient,
        catalog_client: ShopCatalogClient,
        assets_root: Path,
        browser_stream_base_url: str = "",
    ) -> None:
        if not api_token:
            raise ValueError("A non-empty model-training API token is required.")
        self.state_root = Path(state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.api_token = api_token
        self.shop_id = shop_id
        self.live_client = live_client
        self.catalog_client = catalog_client
        self.assets_root = Path(assets_root).resolve()
        self.browser_stream_base_url = browser_stream_base_url.rstrip("/")
        self._mutation_lock = threading.RLock()
        self.store = ModelTrainingStore(self.state_root / "model_training.sqlite")
        self.registrar = CaptureRegistrar(self.state_root, self.store)
        self.exporter = YoloDatasetExporter(self.state_root, self.store)

    def app(self) -> FastAPI:
        app = FastAPI(title="Person Recognition Model Training")
        app.include_router(self.router())
        return app

    def router(self) -> APIRouter:
        router = APIRouter()

        def require_auth(authorization: str | None) -> None:
            if authorization != f"Bearer {self.api_token}":
                raise HTTPException(
                    status_code=401,
                    detail="A valid model-training bearer token is required.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        @router.get("/model-training/")
        def page() -> FileResponse:
            return self._asset("index.html", "text/html")

        @router.get("/model-training/assets/{asset_name}")
        def asset(asset_name: str) -> FileResponse:
            media_types = {"model-training.css": "text/css", "model-training.js": "text/javascript"}
            if asset_name not in media_types:
                raise HTTPException(status_code=404, detail="Unknown model-training asset.")
            return self._asset(asset_name, media_types[asset_name])

        @router.get("/model-training/api/health")
        def health() -> dict[str, Any]:
            return {"status": "ok", "stateRoot": str(self.state_root)}

        @router.get("/model-training/api/cameras")
        def cameras() -> dict[str, Any]:
            try:
                items = self.live_client.cameras()
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            for item in items:
                index = int(item.get("id", item.get("cameraIndex", 0)))
                item["cameraIndex"] = index
                item["cameraNumber"] = index + 1
                item["streamUrl"] = f"{self.browser_stream_base_url}/stream/{index}"
            return {"cameras": items}

        @router.get("/model-training/api/products")
        def products() -> dict[str, Any]:
            return {"products": self.store.list_products()}

        @router.post("/model-training/api/products/refresh")
        def refresh_products(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            require_auth(authorization)
            try:
                products = self.catalog_client.products()
                count = self.store.replace_products(self.shop_id, products)
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"count": count, "products": self.store.list_products()}

        @router.get("/model-training/api/sessions/active")
        def active_session() -> dict[str, Any]:
            session = self.store.active_session()
            if session is None:
                raise HTTPException(status_code=404, detail="No capture session is active.")
            return session

        @router.get("/model-training/api/sessions")
        def sessions(group: str = Query(default="all")) -> dict[str, Any]:
            try:
                return {"sessions": self.store.list_sessions(group=group)}
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        @router.post("/model-training/api/sessions")
        def create_session(
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            require_auth(authorization)
            try:
                camera_index = int(payload["cameraIndex"])
                configured = {int(item.get("id", item.get("cameraIndex", -1))): item for item in self.live_client.cameras()}
                camera = configured.get(camera_index)
                if camera is None:
                    raise ValueError(f"Unknown camera index: {camera_index}")
                return self.store.create_session(
                    shop_id=self.shop_id,
                    product_code=str(payload["productCode"]),
                    scenario=str(payload["scenario"]),
                    camera_index=camera_index,
                    device_id=str(camera["deviceId"]),
                    dataset_intent=str(payload.get("datasetIntent", "development")),
                    notes=None if payload.get("notes") is None else str(payload["notes"]),
                )
            except KeyError as exc:
                raise HTTPException(status_code=422, detail=f"Missing field: {exc.args[0]}") from exc
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        @router.post("/model-training/api/sessions/{session_id}/stop")
        def stop_session(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
            require_auth(authorization)
            try:
                return self.store.stop_session(session_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown capture session.") from exc

        @router.delete("/model-training/api/sessions/active")
        def clear_active_session(
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            require_auth(authorization)
            if payload.get("confirmation") != "CLEAR CURRENT SESSION":
                raise HTTPException(
                    status_code=422,
                    detail="Confirmation must be CLEAR CURRENT SESSION.",
                )
            with self._mutation_lock:
                active = self.store.active_session()
                if active is None:
                    raise HTTPException(status_code=404, detail="No capture session is active.")
                result = self.store.delete_session(str(active["sessionId"]))
                self._delete_owned_files(result["ownedPaths"])
                session_directory = self.registrar.captures_root / str(active["sessionId"])
                try:
                    session_directory.rmdir()
                except OSError:
                    pass
            return {"status": "cleared", **result, "ownedPaths": None}

        @router.delete("/model-training/api/training-data")
        def clear_all_training_data(
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            require_auth(authorization)
            if payload.get("confirmation") != "CLEAR ALL TRAINING DATA":
                raise HTTPException(
                    status_code=422,
                    detail="Confirmation must be CLEAR ALL TRAINING DATA.",
                )
            with self._mutation_lock:
                counts = self.store.clear_training_data()
                self._clear_owned_directories()
            return {"status": "cleared", **counts}

        @router.post("/model-training/api/sessions/{session_id}/captures")
        def capture(
            session_id: str,
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            require_auth(authorization)
            try:
                with self._mutation_lock:
                    session = self.store.get_session(session_id)
                    if session["status"] != "active":
                        raise ValueError("Capture session is not active.")
                    request_id = str(uuid.uuid4())
                    live_capture = self.live_client.capture(int(session["cameraIndex"]))
                    if int(live_capture["cameraIndex"]) != int(session["cameraIndex"]):
                        raise RuntimeError("Live capture camera does not match the session camera.")
                    if str(live_capture["deviceId"]) != str(session["deviceId"]):
                        raise RuntimeError("Live capture device does not match the session device.")
                    return self.registrar.register(session, live_capture, capture_request_id=request_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown session or incomplete live capture response.") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        @router.get("/model-training/api/frames")
        def frames(
            status: str | None = Query(default=None),
            session_id: str | None = Query(default=None, alias="sessionId"),
            exported_only: bool = Query(default=False, alias="exportedOnly"),
            limit: int = Query(default=100, ge=1, le=500),
        ) -> dict[str, Any]:
            return {
                "frames": self.store.list_frames(
                    status=status,
                    session_id=session_id,
                    exported_only=exported_only,
                    limit=limit,
                )
            }

        @router.get("/model-training/api/frames/review-state")
        def review_state(
            session_id: str | None = Query(default=None, alias="sessionId"),
            limit: int = Query(default=200, ge=1, le=500),
        ) -> dict[str, Any]:
            return self.store.frame_queue_state(
                status="needs_review", session_id=session_id, limit=limit
            )

        @router.get("/model-training/api/frames/{frame_id}")
        def frame(frame_id: str) -> dict[str, Any]:
            try:
                return self.store.get_frame(frame_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown frame.") from exc

        @router.get("/model-training/api/frames/{frame_id}/image")
        def frame_image(
            frame_id: str,
            variant: str = Query(default="review"),
            authorization: str | None = Header(default=None),
        ) -> FileResponse:
            if variant == "original":
                require_auth(authorization)
            try:
                path = self.store.frame_image_path(frame_id, variant).resolve(strict=True)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except (KeyError, FileNotFoundError) as exc:
                raise HTTPException(status_code=404, detail="Frame image is unavailable.") from exc
            if not path.is_relative_to(self.state_root):
                raise HTTPException(status_code=403, detail="Frame path is outside model-training storage.")
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

        @router.put("/model-training/api/frames/{frame_id}/annotations")
        def save_annotations(
            frame_id: str,
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            require_auth(authorization)
            try:
                return self.store.save_annotations(frame_id, list(payload.get("boxes", [])))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown frame.") from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        for outcome, path in (
            ("accepted", "accept"),
            ("not_visible", "not-visible"),
            ("rejected", "reject"),
            ("uncertain", "uncertain"),
        ):
            router.add_api_route(
                f"/model-training/api/frames/{{frame_id}}/{path}",
                self._finalizer(outcome, require_auth),
                methods=["POST"],
            )

        @router.post("/model-training/api/datasets")
        def export_dataset(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            require_auth(authorization)
            try:
                with self._mutation_lock:
                    return self.exporter.export()
            except (OSError, RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        return router

    def _delete_owned_files(self, raw_paths: Any) -> None:
        for raw_path in raw_paths:
            path = Path(str(raw_path)).resolve(strict=False)
            if not path.is_relative_to(self.state_root):
                raise RuntimeError(f"Refusing to delete path outside state root: {path}")
            path.unlink(missing_ok=True)

    def _clear_owned_directories(self) -> None:
        for path in (self.state_root / "captures", self.state_root / "datasets"):
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(self.state_root) or resolved == self.state_root:
                raise RuntimeError(f"Refusing to clear unsafe path: {resolved}")
            if resolved.exists():
                shutil.rmtree(resolved)
        self.registrar.ensure_directories()
        self.exporter.datasets_root.mkdir(parents=True, exist_ok=True)

    def _asset(self, name: str, media_type: str) -> FileResponse:
        path = self.assets_root / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Model-training UI is not installed.")
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})

    def _finalizer(self, outcome: str, require_auth: Any) -> Any:
        def finalize(frame_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
            require_auth(authorization)
            try:
                return self.store.finalize_frame(frame_id, outcome)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown frame.") from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return finalize
