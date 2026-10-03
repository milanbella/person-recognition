from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from pipeline.person_photo import PersonPhotoCapture
from pipeline.camera_stats import CameraStats
from pipeline.person_photo_api import create_person_photo_router
from pipeline.observer_api import ObserverCameraSnapshot, observer_snapshot_payload
from pipeline.operator_api import create_operator_router
from pipeline.operator_state import OperatorState
from pipeline.operator_test_store import OperatorTestStore
from pipeline.product_detection import (
    ProductDetection,
    ProductRecognitionResult,
    product_detections_payload,
    product_recognition_payload,
)
from pipeline.product_interaction import (
    ProductInteractionEvent,
    product_interaction_event_payload,
)
from pipeline.shelf_api import (
    ShelfCameraSnapshot,
    shelf_camera_snapshot_payload,
    shelf_status_payload,
)
from pipeline.shelf_proximity import ShelfProximityStatus
from pipeline.shelf_regions import ShelfRegion
from pipeline.visit_registry import is_observer_enabled
from pipeline.world_state import WorldStateProjector
from pipeline.world_state_api import create_world_state_router
from pipeline.world_state_store import WorldStateStore


@dataclass
class _CameraFrame:
    device_id: str
    camera_role: str
    jpeg: bytes | None = None
    sequence: int = 0
    last_frame_monotonic: float | None = None
    observer_snapshot: ObserverCameraSnapshot | None = None
    observer_snapshot_monotonic: float | None = None
    shelf_snapshot: ShelfCameraSnapshot | None = None
    shelf_snapshot_monotonic: float | None = None


class MjpegStreamServer:
    """Publishes processed camera frames through a small MJPEG HTTP API."""

    def __init__(
        self,
        *,
        camera_device_ids: list[str],
        camera_roles: list[str] | None = None,
        host: str = "0.0.0.0",
        port: int = 8002,
        jpeg_quality: int = 70,
        camera_timeout_seconds: float = 3.0,
        enable_mjpeg_streaming: bool = True,
        world_state_db: Path | None = None,
        operator_state_db: Path | None = None,
        operator_runs_root: Path = Path("test-runs"),
        operator_api_token: str | None = None,
        operator_runtime_configuration: dict[str, object] | None = None,
        enable_person_photo_api: bool = False,
        person_photo_api_token: str | None = None,
        shop_opener: Callable[[], Mapping[str, Any]] | None = None,
        shop_shelf_sync_provider: Callable[[int], Mapping[str, Any]] | None = None,
        product_training_capturer: Callable[[int], Mapping[str, Any]] | None = None,
        product_redetector: Callable[
            [bytes, str, tuple[int, int, int, int]],
            tuple[tuple[ProductDetection, ...], bytes, int],
        ]
        | None = None,
        product_interaction_history: tuple[dict[str, Any], ...] = (),
    ) -> None:
        if not camera_device_ids:
            raise ValueError("At least one camera device id is required for streaming.")
        if len(set(camera_device_ids)) != len(camera_device_ids):
            raise ValueError("Camera device ids must be unique.")
        if camera_roles is None:
            camera_roles = ["observer" for _device_id in camera_device_ids]
        if len(camera_roles) != len(camera_device_ids):
            raise ValueError("Camera roles must contain one role per camera device id.")
        if not 1 <= port <= 65535:
            raise ValueError("Stream port must be between 1 and 65535.")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("Stream JPEG quality must be between 1 and 100.")
        if camera_timeout_seconds <= 0:
            raise ValueError("Stream camera timeout must be greater than zero.")

        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.camera_timeout_seconds = camera_timeout_seconds
        self.enable_mjpeg_streaming = enable_mjpeg_streaming
        self._cameras = {
            index: _CameraFrame(device_id=device_id, camera_role=camera_roles[index])
            for index, device_id in enumerate(camera_device_ids)
        }
        self._condition = threading.Condition()
        self.camera_stats = {
            index: CameraStats(device_id) for index, device_id in enumerate(camera_device_ids)
        }
        self._stats_cycle_start: float | None = None
        self._stats_cycle_count = 0
        self._stopping = False
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._server_error: BaseException | None = None
        self._shelf_statuses: tuple[ShelfProximityStatus, ...] = ()
        self._shelf_event_payloads: dict[int, dict[str, object]] = {}
        self._product_payloads_by_visit: dict[int, dict[str, object]] = {}
        self._product_crop_jpegs_by_visit: dict[int, bytes] = {}
        self._product_payloads_by_visit_camera: dict[
            tuple[int, int], dict[str, object]
        ] = {}
        self._product_crop_jpegs_by_visit_camera: dict[tuple[int, int], bytes] = {}
        self._product_source_images_by_visit_camera: dict[tuple[int, int], bytes] = {}
        self._product_payloads_by_camera: dict[int, dict[str, object]] = {}
        self._product_crop_jpegs_by_camera: dict[int, bytes] = {}
        self._product_source_images_by_camera: dict[int, bytes] = {}
        self._product_camera_snapshots: dict[
            str, tuple[dict[str, object], bytes, bytes]
        ] = {}
        self._max_product_camera_snapshots = 100
        self._product_redetector = product_redetector
        self._product_interaction_events = list(product_interaction_history)
        self._current_product_interactions: tuple[dict[str, object], ...] = ()
        self._held_products_by_visit: dict[int, set[str]] = {}
        self._shelf_regions_by_camera: dict[int, dict[int, ShelfRegion]] = {}
        for payload in self._product_interaction_events:
            self._apply_product_interaction_to_held_state(payload)
        effective_world_state_db = (
            operator_state_db if world_state_db is None else world_state_db
        )
        self.world_state_store = (
            None
            if effective_world_state_db is None
            else WorldStateStore(effective_world_state_db)
        )
        self.world_state = (
            None
            if self.world_state_store is None
            else WorldStateProjector(
                camera_device_ids=camera_device_ids,
                camera_roles=camera_roles,
                camera_timeout_seconds=camera_timeout_seconds,
                store=self.world_state_store,
            )
        )
        self.operator_store = (
            None
            if operator_state_db is None
            else OperatorTestStore(
                operator_state_db,
                runs_root=operator_runs_root,
                runtime_configuration=operator_runtime_configuration,
            )
        )
        self.operator_state = (
            None
            if self.operator_store is None
            else OperatorState(
                camera_device_ids=camera_device_ids,
                camera_roles=camera_roles,
                camera_timeout_seconds=camera_timeout_seconds,
                initial_event_id=self.operator_store.max_event_id(),
                event_sink=self.operator_store.enqueue_event,
            )
        )
        self.person_photo_coordinator = (
            PersonPhotoCapture(
                camera_device_ids=camera_device_ids,
            )
            if enable_person_photo_api
            else None
        )
        self.app = self._build_app()
        if self.person_photo_coordinator is not None:
            if not person_photo_api_token:
                raise ValueError("person_photo_api_token is required when person photo is enabled.")
            self.app.include_router(
                create_person_photo_router(
                    coordinator=self.person_photo_coordinator,
                    api_token=person_photo_api_token,
                    operator_api_token=operator_api_token,
                )
            )
        if self.world_state_store is not None:
            assert self.world_state is not None
            self.app.include_router(
                create_world_state_router(
                    projector=self.world_state,
                    store=self.world_state_store,
                    operator_store=self.operator_store,
                )
            )
        if self.operator_store is not None:
            assert self.operator_state is not None
            self.app.include_router(
                create_operator_router(
                    state=self.operator_state,
                    store=self.operator_store,
                    api_token=operator_api_token,
                    world_state_store=self.world_state_store,
                    assets_root=Path(__file__).resolve().parent.parent
                    / "operator_console",
                    shop_opener=shop_opener,
                    shop_shelf_sync_provider=shop_shelf_sync_provider,
                    product_training_capturer=product_training_capturer,
                    person_photo_available=(
                        self.person_photo_coordinator is not None
                    ),
                )
            )

    def record_processing_cycle(self) -> None:
        """Called by the live loop, matching checkout's one-second loop FPS."""
        now = time.monotonic()
        if self._stats_cycle_start is None:
            self._stats_cycle_start = now
        self._stats_cycle_count += 1
        elapsed = now - self._stats_cycle_start
        if elapsed >= 1.0:
            fps = self._stats_cycle_count / elapsed
            for stats in self.camera_stats.values():
                stats.set_processing_fps(fps)
            self._stats_cycle_start = now
            self._stats_cycle_count = 0

    def person_photo_active(self) -> bool:
        return (
            self.person_photo_coordinator is not None
            and self.person_photo_coordinator.has_pending_requests()
        )

    def update_person_photo_visit_state(
        self,
        visits: Mapping[int, tuple[str, str | None, str]],
    ) -> None:
        if self.person_photo_coordinator is not None:
            self.person_photo_coordinator.update_visit_state(visits)

    def observe_person_photo_frame(
        self,
        *,
        camera_index: int,
        device_id: str,
        rgb_sequence_number: int,
        observed_at_unix_milliseconds: int,
        frame: np.ndarray,
        tracks: tuple[Any, ...] | list[Any],
        assignments: Mapping[int, Any],
        customer_ids_by_visit: Mapping[int, str],
    ) -> None:
        if self.person_photo_coordinator is None:
            return
        self.person_photo_coordinator.observe_frame(
            camera_index=camera_index,
            device_id=device_id,
            rgb_sequence_number=rgb_sequence_number,
            observed_at_unix_milliseconds=observed_at_unix_milliseconds,
            frame=frame,
            tracks=tracks,
            assignments=assignments,
            customer_ids_by_visit=customer_ids_by_visit,
        )

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/stream/{cam_index}")
        def stream(cam_index: int) -> StreamingResponse:
            if cam_index not in self._cameras:
                raise HTTPException(status_code=404, detail=f"Camera index {cam_index} is not configured.")
            if not self.enable_mjpeg_streaming:
                raise HTTPException(status_code=503, detail="MJPEG streaming is disabled.")
            return StreamingResponse(
                self._generate_stream(cam_index),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        @app.get("/cameras-status")
        def cameras_status() -> dict[str, list[dict[str, int | str]]]:
            return self.camera_status_payload()

        @app.get("/stats/{cam_index}")
        async def camera_stats(cam_index: int, request: Request) -> StreamingResponse:
            if cam_index < 0:
                raise HTTPException(status_code=400, detail="Camera index cannot be negative.")
            if cam_index not in self.camera_stats:
                raise HTTPException(status_code=404, detail="Camera index is not configured.")

            async def generate():
                while not self._stopping and not await request.is_disconnected():
                    yield json.dumps(self.camera_stats[cam_index].payload(cam_index)) + "\n"
                    await asyncio.sleep(2)

            return StreamingResponse(
                generate(), media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.get("/observer-cameras/{cam_index}/observations")
        def observer_camera_observations(cam_index: int) -> dict[str, object]:
            if cam_index not in self._cameras:
                raise HTTPException(status_code=404, detail=f"Camera index {cam_index} is not configured.")
            if not is_observer_enabled(self._cameras[cam_index].camera_role):
                raise HTTPException(
                    status_code=409,
                    detail=f"Camera index {cam_index} is not observer-capable.",
                )
            return self.observer_snapshot_payload(cam_index)

        @app.get("/observer-cameras/{cam_index}/shelves")
        def observer_camera_shelves(cam_index: int) -> dict[str, object]:
            if cam_index not in self._cameras:
                raise HTTPException(status_code=404, detail=f"Camera index {cam_index} is not configured.")
            if not is_observer_enabled(self._cameras[cam_index].camera_role):
                raise HTTPException(
                    status_code=409,
                    detail=f"Camera index {cam_index} is not observer-capable.",
                )
            return self.shelf_snapshot_payload(cam_index)

        @app.get("/shelves/status")
        def shelves_status() -> dict[str, object]:
            with self._condition:
                statuses = self._shelf_statuses
            return shelf_status_payload(statuses)

        @app.get("/shelf-regions/cameras/{camera_index}")
        def camera_shelf_regions(camera_index: int) -> dict[str, object]:
            if camera_index not in self._cameras:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera index {camera_index} is not configured.",
                )
            with self._condition:
                regions = self._shelf_regions_by_camera.get(camera_index, {})
                device_id = self._cameras[camera_index].device_id
            return {
                "cameraIndex": camera_index,
                "cameraNumber": camera_index + 1,
                "deviceId": device_id,
                "regions": [
                    {
                        "shelfId": shelf_id,
                        "polygonNormalized": [
                            {"x": point.x, "y": point.y}
                            for point in region.polygon
                        ],
                        "polygonsNormalized": [
                            [{"x": point.x, "y": point.y} for point in polygon]
                            for polygon in region.polygons
                        ],
                        "depthToleranceMm": region.depth_tolerance_mm,
                        "offShelfHysteresisMm": region.off_shelf_hysteresis_mm,
                        "has3dDepthModel": region.depth_model is not None,
                        "depthModel": (
                            None
                            if region.depth_model is None
                            else {
                                "columns": region.depth_model.columns,
                                "rows": region.depth_model.rows,
                                "alignedDepthWidth": region.depth_model.aligned_depth_width,
                                "alignedDepthHeight": region.depth_model.aligned_depth_height,
                                "validCellCount": len(region.depth_model.cells),
                                "calibrationFrameCount": region.depth_model.calibration_frame_count,
                                "calibratedAtUnixMilliseconds": region.depth_model.calibrated_at_unix_milliseconds,
                                "cameraCalibrationId": region.depth_model.camera_calibration_id,
                            }
                        ),
                    }
                    for shelf_id, region in sorted(regions.items())
                ],
            }

        @app.get("/product-interaction-events")
        def product_interaction_events(
            afterEventId: int = 0,
            limit: int = 100,
        ) -> dict[str, object]:
            return self.product_interaction_events_payload(
                after_id=afterEventId,
                limit=limit,
            )

        @app.get("/product-interactions/current")
        def current_product_interactions() -> dict[str, object]:
            with self._condition:
                interactions = copy.deepcopy(self._current_product_interactions)
            return {"interactions": interactions, "authority": "diagnostic"}

        @app.get("/pose-observations/cameras/{camera_index}")
        def camera_pose_observations(camera_index: int) -> dict[str, object]:
            payload = self.observer_snapshot_payload(camera_index)
            return {
                "camera": payload["camera"],
                "frame": payload["frame"],
                "poses": [
                    {
                        "trackId": observation["trackId"],
                        "visitId": observation["visitId"],
                        "pose": observation["pose"],
                    }
                    for observation in payload.get("observations", [])
                    if observation.get("pose") is not None
                ],
            }

        @app.get("/world-state/visits/{visit_id}/held-products")
        def held_products(visit_id: int) -> dict[str, object]:
            with self._condition:
                labels = sorted(self._held_products_by_visit.get(visit_id, set()))
            return {
                "visitId": visit_id,
                "products": labels,
                "count": len(labels),
                "authority": "candidate",
            }

        @app.get("/world-state/visits/{visit_id}/product-crop.jpg")
        def visit_product_crop(visit_id: int) -> Response:
            with self._condition:
                jpeg = self._product_crop_jpegs_by_visit.get(visit_id)
            if jpeg is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No product crop is available for visit {visit_id}.",
                )
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/world-state/visits/{visit_id}/product-observations")
        def visit_product_observations(visit_id: int) -> dict[str, object]:
            return self.product_camera_observations_payload(visit_id)

        @app.get(
            "/world-state/visits/{visit_id}/product-observations/"
            "{camera_index}/crop.jpg"
        )
        def visit_camera_product_crop(visit_id: int, camera_index: int) -> Response:
            if camera_index not in self._cameras:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera index {camera_index} is not configured.",
                )
            with self._condition:
                evidence = self._latest_camera_product_evidence_locked(
                    visit_id,
                    camera_index,
                )
                jpeg = None if evidence is None else evidence[1]
            if jpeg is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No product crop is available for visit "
                        f"{visit_id} on camera {camera_index}."
                    ),
                )
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.post(
            "/world-state/visits/{visit_id}/product-observations/"
            "{camera_index}/snapshot"
        )
        def create_visit_camera_product_snapshot(
            visit_id: int,
            camera_index: int,
        ) -> dict[str, object]:
            if camera_index not in self._cameras:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera index {camera_index} is not configured.",
                )
            try:
                return self.create_product_camera_snapshot(
                    visit_id=visit_id,
                    camera_index=camera_index,
                )
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error

        @app.get("/product-observations/cameras")
        def latest_camera_product_observations() -> dict[str, object]:
            return self.latest_product_camera_observations_payload()

        @app.get("/product-observations/cameras/{camera_index}/crop.jpg")
        def latest_camera_product_crop(camera_index: int) -> Response:
            if camera_index not in self._cameras:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera index {camera_index} is not configured.",
                )
            with self._condition:
                evidence = self._latest_product_evidence_for_camera_locked(
                    camera_index
                )
                jpeg = None if evidence is None else evidence[1]
            if jpeg is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No product crop is available for camera "
                        f"{camera_index}."
                    ),
                )
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.post("/product-observations/cameras/{camera_index}/snapshot")
        def create_latest_camera_product_snapshot(
            camera_index: int,
        ) -> dict[str, object]:
            try:
                return self.create_latest_product_camera_snapshot(camera_index)
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error

        @app.get("/product-observation-snapshots/{snapshot_id}/crop.jpg")
        def product_camera_snapshot_crop(snapshot_id: str) -> Response:
            with self._condition:
                snapshot = self._product_camera_snapshots.get(snapshot_id)
                jpeg = None if snapshot is None else snapshot[1]
            if jpeg is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Product snapshot {snapshot_id} is not available.",
                )
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=31536000, immutable"},
            )

        @app.post("/product-observation-snapshots/{snapshot_id}/detect")
        def redetect_product_camera_snapshot(snapshot_id: str) -> dict[str, object]:
            if self._product_redetector is None:
                raise HTTPException(
                    status_code=503,
                    detail="Product redetection is not enabled.",
                )
            try:
                return self.redetect_product_camera_snapshot(snapshot_id)
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            except (RuntimeError, ValueError) as error:
                raise HTTPException(status_code=500, detail=str(error)) from error

        @app.get("/shelf-events")
        def shelf_events(
            afterEventId: int = 0,
            limit: int = 100,
        ) -> dict[str, object]:
            if afterEventId < 0:
                raise HTTPException(status_code=422, detail="afterEventId must not be negative.")
            if not 1 <= limit <= 1000:
                raise HTTPException(status_code=422, detail="limit must be between 1 and 1000.")
            with self._condition:
                payloads = [
                    payload
                    for event_id, payload in sorted(self._shelf_event_payloads.items())
                    if event_id > afterEventId
                ][:limit]
            return {
                "events": payloads,
                "lastEventId": (
                    afterEventId
                    if not payloads
                    else int(payloads[-1]["eventId"])
                ),
            }

        return app

    def start(self, startup_timeout_seconds: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("MJPEG stream server has already been started.")

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def run_server() -> None:
            try:
                assert self._server is not None
                self._server.run()
            except BaseException as exc:
                self._server_error = exc

        self._thread = threading.Thread(
            target=run_server,
            name="person-recognition-mjpeg-server",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._server.started:
                api_name = (
                    "MJPEG streaming API"
                    if self.enable_mjpeg_streaming
                    else "Operator/world-state API"
                )
                print(f"{api_name} listening on http://{self.host}:{self.port}")
                return
            if not self._thread.is_alive():
                error = self._server_error or RuntimeError("Uvicorn stopped during startup.")
                raise RuntimeError(f"Could not start MJPEG streaming API on {self.host}:{self.port}.") from error
            time.sleep(0.01)

        self.stop()
        raise TimeoutError(f"Timed out starting MJPEG streaming API on {self.host}:{self.port}.")

    def publish(
        self,
        camera_index: int,
        frame: np.ndarray,
        *,
        rgb_sequence_number: int | None = None,
    ) -> None:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        if not self.enable_mjpeg_streaming:
            raise RuntimeError("MJPEG frame publication is disabled.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("MJPEG frames must be BGR images with shape (height, width, 3).")

        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded:
            raise RuntimeError(f"JPEG encoding failed for camera index {camera_index}.")

        with self._condition:
            camera = self._cameras[camera_index]
            camera.jpeg = jpeg.tobytes()
            camera.sequence += 1
            camera.last_frame_monotonic = time.monotonic()
            stream_sequence = camera.sequence
            jpeg_bytes = camera.jpeg
            self._condition.notify_all()
        assert jpeg_bytes is not None
        if self.operator_state is not None:
            self.operator_state.mark_camera_frame(
                camera_index,
                jpeg=jpeg_bytes,
                stream_sequence=stream_sequence,
                rgb_sequence_number=rgb_sequence_number,
            )
        if self.world_state is not None:
            self.world_state.mark_camera_frame(
                camera_index,
                rgb_sequence_number=rgb_sequence_number,
            )

    def camera_status_payload(self) -> dict[str, list[dict[str, int | str]]]:
        now = time.monotonic()
        with self._condition:
            cameras = []
            for index, camera in self._cameras.items():
                last_frame = self.camera_stats[index].last_received_monotonic
                # Replay/standalone publishers may not report raw packet receipt.
                if last_frame is None:
                    last_frame = camera.last_frame_monotonic
                cameras.append({
                    "id": index,
                    "deviceId": camera.device_id,
                    "status": (
                        "active"
                        if last_frame is not None
                        and now - last_frame <= self.camera_timeout_seconds
                        else "offline"
                    ),
                })
        return {"cameras": cameras}

    def publish_observer_snapshot(
        self,
        camera_index: int,
        snapshot: ObserverCameraSnapshot,
    ) -> None:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        if not is_observer_enabled(self._cameras[camera_index].camera_role):
            raise ValueError(f"Camera index {camera_index} is not observer-capable.")
        if snapshot.camera_index != camera_index:
            raise ValueError("Observer snapshot camera index does not match publication index.")

        with self._condition:
            camera = self._cameras[camera_index]
            camera.observer_snapshot = snapshot
            camera.observer_snapshot_monotonic = time.monotonic()
        if self.operator_state is not None:
            self.operator_state.publish_observer_snapshot(snapshot)
        if self.world_state is not None:
            self.world_state.publish_observer_snapshot(snapshot)

    def publish_product_recognition(
        self,
        result: ProductRecognitionResult,
        *,
        visit_id: int,
        customer_id: str | None,
        max_age_seconds: float,
    ) -> None:
        payload = product_recognition_payload(
            result,
            visit_id=visit_id,
            customer_id=customer_id,
            max_age_seconds=max_age_seconds,
        )
        with self._condition:
            self._product_payloads_by_visit[visit_id] = payload
            self._product_crop_jpegs_by_visit[visit_id] = result.crop_jpeg
            camera_key = (visit_id, result.camera_index)
            self._product_payloads_by_visit_camera[camera_key] = payload
            self._product_crop_jpegs_by_visit_camera[camera_key] = result.crop_jpeg
            self._product_source_images_by_visit_camera[camera_key] = (
                result.source_crop_image or result.crop_jpeg
            )
        if self.world_state is not None:
            self.world_state.publish_product_recognition(payload)

    def publish_product_interaction_event(
        self,
        event: ProductInteractionEvent,
        *,
        persisted_id: int,
        status: str = "candidate",
    ) -> None:
        payload = product_interaction_event_payload(event, status=status)
        payload["id"] = persisted_id
        with self._condition:
            self._product_interaction_events.append(payload)
            self._product_interaction_events = self._product_interaction_events[-1000:]
            self._apply_product_interaction_to_held_state(payload)

    def publish_shelf_regions(
        self,
        camera_index: int,
        regions: Mapping[int, ShelfRegion],
    ) -> None:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        with self._condition:
            self._shelf_regions_by_camera[camera_index] = dict(regions)

    def publish_current_product_interactions(
        self,
        interactions: tuple[dict[str, object], ...],
    ) -> None:
        with self._condition:
            self._current_product_interactions = copy.deepcopy(interactions)

    def product_interaction_events_payload(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
    ) -> dict[str, object]:
        safe_limit = min(500, max(1, limit))
        with self._condition:
            events = [
                copy.deepcopy(payload)
                for payload in self._product_interaction_events
                if int(payload.get("id", 0)) > after_id
            ][:safe_limit]
        return {
            "events": events,
            "nextEventId": after_id if not events else int(events[-1]["id"]),
        }

    def _apply_product_interaction_to_held_state(
        self,
        payload: Mapping[str, object],
    ) -> None:
        visit_id = int(payload["visitId"])
        label = str(payload["productLabel"])
        held = self._held_products_by_visit.setdefault(visit_id, set())
        if payload["eventType"] == "PRODUCT_PICKED":
            held.add(label)
        elif payload["eventType"] == "PRODUCT_RETURNED":
            held.discard(label)

    def publish_camera_product_recognition(
        self,
        result: ProductRecognitionResult,
        *,
        max_age_seconds: float,
    ) -> None:
        payload = product_recognition_payload(
            result,
            visit_id=None,
            customer_id=None,
            max_age_seconds=max_age_seconds,
        )
        with self._condition:
            self._product_payloads_by_camera[result.camera_index] = payload
            self._product_crop_jpegs_by_camera[result.camera_index] = (
                result.crop_jpeg
            )
            self._product_source_images_by_camera[result.camera_index] = (
                result.source_crop_image or result.crop_jpeg
            )

    def product_recognition_payload(self, visit_id: int) -> dict[str, object]:
        now_ms = time.time_ns() // 1_000_000
        with self._condition:
            payload = copy.deepcopy(self._product_payloads_by_visit.get(visit_id))
        if payload is None:
            return {
                "status": "unknown",
                "freshness": "unknown",
                "visitId": visit_id,
                "bestCandidate": None,
                "candidates": [],
            }
        observed_ms = int(payload["observedAtUnixMilliseconds"])
        age_ms = max(0, now_ms - observed_ms)
        payload["ageMilliseconds"] = age_ms
        payload["freshness"] = (
            "current"
            if age_ms <= int(payload["maxAgeMilliseconds"])
            else "stale"
        )
        return payload

    def product_camera_observations_payload(
        self,
        visit_id: int,
    ) -> dict[str, object]:
        now_ms = time.time_ns() // 1_000_000
        with self._condition:
            observations: dict[int, dict[str, object]] = {}
            for camera_index in self._cameras:
                evidence = self._latest_camera_product_evidence_locked(
                    visit_id,
                    camera_index,
                )
                if evidence is not None:
                    observations[camera_index] = copy.deepcopy(evidence[0])

        cameras: list[dict[str, object]] = []
        for camera_index, camera in self._cameras.items():
            payload = observations.get(camera_index)
            if payload is None:
                cameras.append(
                    {
                        "cameraIndex": camera_index,
                        "deviceId": camera.device_id,
                        "cameraRole": camera.camera_role,
                        "visitId": visit_id,
                        "status": "unknown",
                        "freshness": "unknown",
                        "cropAvailable": False,
                        "bestCandidate": None,
                        "candidates": [],
                    }
                )
                continue
            observed_ms = int(payload["observedAtUnixMilliseconds"])
            age_ms = max(0, now_ms - observed_ms)
            payload["ageMilliseconds"] = age_ms
            payload["freshness"] = (
                "current"
                if age_ms <= int(payload["maxAgeMilliseconds"])
                else "stale"
            )
            payload["cameraRole"] = camera.camera_role
            payload["cropAvailable"] = True
            cameras.append(payload)
        return {
            "visitId": visit_id,
            "generatedAtUnixMilliseconds": now_ms,
            "cameras": cameras,
        }

    def latest_product_camera_observations_payload(self) -> dict[str, object]:
        now_ms = time.time_ns() // 1_000_000
        with self._condition:
            observations = {
                camera_index: copy.deepcopy(evidence[0])
                for camera_index in self._cameras
                if (
                    evidence := self._latest_product_evidence_for_camera_locked(
                        camera_index
                    )
                )
                is not None
            }

        cameras: list[dict[str, object]] = []
        for camera_index, camera in self._cameras.items():
            payload = observations.get(camera_index)
            if payload is None:
                cameras.append(
                    {
                        "cameraIndex": camera_index,
                        "deviceId": camera.device_id,
                        "cameraRole": camera.camera_role,
                        "visitId": None,
                        "status": "unknown",
                        "freshness": "unknown",
                        "cropAvailable": False,
                        "bestCandidate": None,
                        "candidates": [],
                    }
                )
                continue
            observed_ms = int(payload["observedAtUnixMilliseconds"])
            age_ms = max(0, now_ms - observed_ms)
            payload["ageMilliseconds"] = age_ms
            payload["freshness"] = (
                "current"
                if age_ms <= int(payload["maxAgeMilliseconds"])
                else "stale"
            )
            payload["cameraRole"] = camera.camera_role
            payload["cropAvailable"] = True
            cameras.append(payload)
        return {
            "scope": "latest_by_camera",
            "generatedAtUnixMilliseconds": now_ms,
            "cameras": cameras,
        }

    def create_product_camera_snapshot(
        self,
        *,
        visit_id: int,
        camera_index: int,
    ) -> dict[str, object]:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        with self._condition:
            evidence = self._latest_camera_product_evidence_locked(
                visit_id,
                camera_index,
            )
            if evidence is None:
                raise KeyError(
                    "No product evidence is available for visit "
                    f"{visit_id} on camera {camera_index}."
                )
            source_image = self._source_product_crop_for_payload_locked(
                evidence[0], camera_index
            )
        return self._create_product_camera_snapshot_from_evidence(
            camera_index=camera_index,
            evidence=evidence,
            source_image=source_image,
            requested_visit_id=visit_id,
        )

    def create_latest_product_camera_snapshot(
        self,
        camera_index: int,
    ) -> dict[str, object]:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        with self._condition:
            evidence = self._latest_product_evidence_for_camera_locked(camera_index)
            if evidence is None:
                raise KeyError(
                    f"No product evidence is available on camera {camera_index}."
                )
            source_image = self._source_product_crop_for_payload_locked(
                evidence[0], camera_index
            )
        requested_visit_id = evidence[0].get("visitId")
        return self._create_product_camera_snapshot_from_evidence(
            camera_index=camera_index,
            evidence=evidence,
            source_image=source_image,
            requested_visit_id=(
                int(requested_visit_id)
                if requested_visit_id is not None
                else None
            ),
        )

    def _create_product_camera_snapshot_from_evidence(
        self,
        *,
        camera_index: int,
        evidence: tuple[dict[str, object], bytes],
        source_image: bytes,
        requested_visit_id: int | None,
    ) -> dict[str, object]:
        snapshot_id = uuid.uuid4().hex
        now_ms = time.time_ns() // 1_000_000
        captured_ms = int(evidence[0].get("frozenAtUnixMilliseconds", now_ms))
        with self._condition:
            payload = copy.deepcopy(evidence[0])
            jpeg = evidence[1]
            payload["frozenAtUnixMilliseconds"] = captured_ms
            observed_ms = int(payload["observedAtUnixMilliseconds"])
            age_ms = max(0, now_ms - observed_ms)
            payload["ageMilliseconds"] = age_ms
            payload["freshness"] = (
                "current"
                if age_ms <= int(payload["maxAgeMilliseconds"])
                else "stale"
            )
            payload["cameraRole"] = self._cameras[camera_index].camera_role
            payload["cropAvailable"] = True
            self._product_camera_snapshots[snapshot_id] = (
                payload,
                jpeg,
                source_image,
            )
            while len(self._product_camera_snapshots) > self._max_product_camera_snapshots:
                oldest_snapshot_id = next(iter(self._product_camera_snapshots))
                del self._product_camera_snapshots[oldest_snapshot_id]
        return {
            "snapshotId": snapshot_id,
            "requestedVisitId": requested_visit_id,
            "capturedAtUnixMilliseconds": captured_ms,
            "camera": payload,
            "imageUrl": f"/product-observation-snapshots/{snapshot_id}/crop.jpg",
        }

    def redetect_product_camera_snapshot(
        self,
        snapshot_id: str,
    ) -> dict[str, object]:
        if self._product_redetector is None:
            raise RuntimeError("Product redetection is not enabled.")
        with self._condition:
            snapshot = self._product_camera_snapshots.get(snapshot_id)
            if snapshot is None:
                raise KeyError(f"Product snapshot {snapshot_id} is not available.")
            original_payload = copy.deepcopy(snapshot[0])
            source_image = snapshot[2]
        person_box_payload = original_payload["personBoxInCrop"]
        if not isinstance(person_box_payload, dict):
            raise ValueError("Frozen product snapshot has no person crop bounds.")
        person_box = tuple(
            int(person_box_payload[key]) for key in ("x1", "y1", "x2", "y2")
        )
        detections, annotated_jpeg, inference_ms = self._product_redetector(
            source_image,
            str(original_payload["scope"]),
            person_box,
        )
        candidates = product_detections_payload(detections)
        original_payload["status"] = "recognized" if candidates else "no_product"
        original_payload["bestCandidate"] = None if not candidates else candidates[0]
        original_payload["candidates"] = candidates
        original_payload["inferenceMilliseconds"] = inference_ms
        original_payload["redetectedAtUnixMilliseconds"] = (
            time.time_ns() // 1_000_000
        )
        requested_visit_id = original_payload.get("visitId")
        return self._create_product_camera_snapshot_from_evidence(
            camera_index=int(original_payload["cameraIndex"]),
            evidence=(original_payload, annotated_jpeg),
            source_image=source_image,
            requested_visit_id=(
                int(requested_visit_id)
                if requested_visit_id is not None
                else None
            ),
        )

    def _source_product_crop_for_payload_locked(
        self,
        payload: dict[str, object],
        camera_index: int,
    ) -> bytes:
        visit_id = payload.get("visitId")
        if visit_id is not None:
            source = self._product_source_images_by_visit_camera.get(
                (int(visit_id), camera_index)
            )
            if source is not None:
                return source
        source = self._product_source_images_by_camera.get(camera_index)
        if source is not None:
            return source
        raise KeyError(f"No source product crop is available on camera {camera_index}.")

    def _latest_camera_product_evidence_locked(
        self,
        visit_id: int,
        camera_index: int,
    ) -> tuple[dict[str, object], bytes] | None:
        candidates: list[tuple[dict[str, object], bytes]] = []
        visit_payload = self._product_payloads_by_visit_camera.get(
            (visit_id, camera_index)
        )
        visit_jpeg = self._product_crop_jpegs_by_visit_camera.get(
            (visit_id, camera_index)
        )
        if visit_payload is not None and visit_jpeg is not None:
            candidates.append((visit_payload, visit_jpeg))
        camera_payload = self._product_payloads_by_camera.get(camera_index)
        camera_jpeg = self._product_crop_jpegs_by_camera.get(camera_index)
        if camera_payload is not None and camera_jpeg is not None:
            candidates.append((camera_payload, camera_jpeg))
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: int(item[0]["observedAtUnixMilliseconds"]),
        )

    def _latest_product_evidence_for_camera_locked(
        self,
        camera_index: int,
    ) -> tuple[dict[str, object], bytes] | None:
        candidates: list[tuple[dict[str, object], bytes]] = []
        camera_payload = self._product_payloads_by_camera.get(camera_index)
        camera_jpeg = self._product_crop_jpegs_by_camera.get(camera_index)
        if camera_payload is not None and camera_jpeg is not None:
            candidates.append((camera_payload, camera_jpeg))
        for (visit_id, candidate_camera_index), payload in (
            self._product_payloads_by_visit_camera.items()
        ):
            if candidate_camera_index != camera_index:
                continue
            jpeg = self._product_crop_jpegs_by_visit_camera.get(
                (visit_id, candidate_camera_index)
            )
            if jpeg is not None:
                candidates.append((payload, jpeg))
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: int(item[0]["observedAtUnixMilliseconds"]),
        )

    def observer_snapshot_payload(self, camera_index: int) -> dict[str, object]:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")

        now = time.monotonic()
        with self._condition:
            camera = self._cameras[camera_index]
            if not is_observer_enabled(camera.camera_role):
                raise ValueError(f"Camera index {camera_index} is not observer-capable.")
            snapshot = camera.observer_snapshot
            published_monotonic = camera.observer_snapshot_monotonic

        if snapshot is None or published_monotonic is None:
            return {
                "camera": {
                    "id": camera_index,
                    "deviceId": camera.device_id,
                    "role": camera.camera_role,
                    "status": "starting",
                },
                "frame": None,
                "observations": [],
            }

        age_milliseconds = max(0, int(round((now - published_monotonic) * 1000.0)))
        is_fresh = now - published_monotonic <= self.camera_timeout_seconds
        return observer_snapshot_payload(
            snapshot,
            age_milliseconds=age_milliseconds,
            status="active" if is_fresh else "offline",
            include_observations=is_fresh,
        )

    def publish_shelf_snapshot(
        self,
        camera_index: int,
        snapshot: ShelfCameraSnapshot,
    ) -> None:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        if not is_observer_enabled(self._cameras[camera_index].camera_role):
            raise ValueError(f"Camera index {camera_index} is not observer-capable.")
        if snapshot.camera_index != camera_index:
            raise ValueError("Shelf snapshot camera index does not match publication index.")
        with self._condition:
            camera = self._cameras[camera_index]
            camera.shelf_snapshot = snapshot
            camera.shelf_snapshot_monotonic = time.monotonic()
        if self.operator_state is not None:
            self.operator_state.publish_shelf_snapshot(snapshot)
        if self.world_state is not None:
            self.world_state.publish_shelf_snapshot(snapshot)

    def shelf_snapshot_payload(self, camera_index: int) -> dict[str, object]:
        if camera_index not in self._cameras:
            raise KeyError(f"Camera index {camera_index} is not configured.")
        now = time.monotonic()
        with self._condition:
            camera = self._cameras[camera_index]
            if not is_observer_enabled(camera.camera_role):
                raise ValueError(f"Camera index {camera_index} is not observer-capable.")
            snapshot = camera.shelf_snapshot
            published_monotonic = camera.shelf_snapshot_monotonic
        if snapshot is None or published_monotonic is None:
            return {
                "camera": {
                    "id": camera_index,
                    "deviceId": camera.device_id,
                    "role": camera.camera_role,
                    "status": "starting",
                },
                "frame": None,
                "shelves": [],
            }
        age_milliseconds = max(
            0,
            int(round((now - published_monotonic) * 1000.0)),
        )
        is_fresh = now - published_monotonic <= self.camera_timeout_seconds
        return shelf_camera_snapshot_payload(
            snapshot,
            age_milliseconds=age_milliseconds,
            status="active" if is_fresh else "offline",
            include_observations=is_fresh,
        )

    def publish_shelf_statuses(
        self,
        statuses: tuple[ShelfProximityStatus, ...],
    ) -> None:
        with self._condition:
            self._shelf_statuses = tuple(statuses)
        if self.operator_state is not None:
            self.operator_state.publish_shelf_statuses(statuses)
        if self.world_state is not None:
            self.world_state.publish_shelf_statuses(statuses)

    def publish_shelf_event_payloads(
        self,
        payloads: list[dict[str, object]],
        *,
        emit_operator_events: bool = True,
    ) -> None:
        with self._condition:
            for payload in payloads:
                event_id = payload.get("eventId")
                if not isinstance(event_id, int):
                    raise ValueError("Shelf event payload must contain an integer eventId.")
                self._shelf_event_payloads[event_id] = dict(payload)
            if len(self._shelf_event_payloads) > 1000:
                retained_ids = sorted(self._shelf_event_payloads)[-1000:]
                self._shelf_event_payloads = {
                    event_id: self._shelf_event_payloads[event_id]
                    for event_id in retained_ids
                }
        if emit_operator_events and self.operator_state is not None:
            for payload in payloads:
                camera = payload.get("camera")
                camera_payload = camera if isinstance(camera, dict) else {}
                self.operator_state.publish_event(
                    event_type=str(payload.get("eventType", "shelf_event")),
                    occurred_at_unix_milliseconds=int(
                        payload.get(
                            "occurredAtUnixMilliseconds",
                            time.time_ns() // 1_000_000,
                        )
                    ),
                    host_synced_seconds=(
                        None
                        if payload.get("hostSyncedSeconds") is None
                        else float(payload["hostSyncedSeconds"])
                    ),
                    camera_index=(
                        None
                        if camera_payload.get("id") is None
                        else int(camera_payload["id"])
                    ),
                    device_id=(
                        None
                        if camera_payload.get("deviceId") is None
                        else str(camera_payload["deviceId"])
                    ),
                    rgb_sequence_number=(
                        None
                        if payload.get("rgbSequenceNumber") is None
                        else int(payload["rgbSequenceNumber"])
                    ),
                    track_id=(
                        None
                        if camera_payload.get("trackId") is None
                        else int(camera_payload["trackId"])
                    ),
                    visit_id=(
                        None
                        if payload.get("visitId") is None
                        else int(payload["visitId"])
                    ),
                    payload=payload,
                )
        if self.world_state is not None:
            for payload in payloads:
                self.world_state.publish_transition(
                    event_type=str(payload.get("eventType", "shelf_event")),
                    entity_type="shelf",
                    entity_id=int(payload.get("shelfId", 0)),
                    payload=payload,
                    occurred_at_unix_milliseconds=int(
                        payload.get("occurredAtUnixMilliseconds", time.time_ns() // 1_000_000)
                    ),
                    host_synced_seconds=(
                        None
                        if payload.get("hostSyncedSeconds") is None
                        else float(payload["hostSyncedSeconds"])
                    ),
                    source_event_id=(
                        None if payload.get("eventId") is None else int(payload["eventId"])
                    ),
                )

    def publish_entrance_event(
        self,
        *,
        event_type: str,
        camera_index: int,
        payload: dict[str, object],
    ) -> None:
        if self.operator_state is None and self.world_state is None:
            return
        visit_id = payload.get("visit_id")
        track_id = int(payload["track_id"])
        host_seconds = float(payload["host_synced_seconds"])
        device_id = str(payload["device_id"])
        status = "inside" if event_type == "entry_accepted" else "left"
        if self.operator_state is not None:
            self.operator_state.publish_visit_state(
                None if visit_id is None else int(visit_id),
                status=status,
                origin="entrance_confirmed",
                host_synced_seconds=host_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
            )
            self.operator_state.publish_event(
                event_type=event_type,
                occurred_at_unix_milliseconds=time.time_ns() // 1_000_000,
                host_synced_seconds=host_seconds,
                camera_index=camera_index,
                device_id=device_id,
                rgb_sequence_number=int(payload["rgb_sequence_num"]),
                track_id=track_id,
                visit_id=None if visit_id is None else int(visit_id),
                payload=payload,
            )
        if self.world_state is not None:
            self.world_state.publish_visit_state(
                None if visit_id is None else int(visit_id),
                status=status,
                origin="entrance_confirmed",
                host_synced_seconds=host_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
                event_type=event_type,
            )

    def save_plane_crossing_evidence(
        self,
        *,
        filename_stem: str,
        image: np.ndarray,
        jpeg_quality: int,
        metadata: Mapping[str, Any],
    ) -> str | None:
        if self.operator_store is None or self.operator_store.active_run() is None:
            return None
        encoded, buffer = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not encoded:
            raise RuntimeError("Failed to encode plane-crossing evidence JPEG.")
        return self.operator_store.save_plane_crossing_evidence(
            filename_stem=filename_stem,
            jpeg=buffer.tobytes(),
            metadata=metadata,
        )

    def publish_shop_api_leave_result(
        self,
        *,
        event_type: str,
        camera_index: int,
        device_id: str,
        track_id: int,
        visit_id: int | None,
        host_synced_seconds: float,
        payload: dict[str, object],
    ) -> None:
        if event_type not in {"shop_leave_persisted", "shop_leave_persist_failed"}:
            raise ValueError(f"Unsupported shop leave result event: {event_type}")
        occurred_at_unix_milliseconds = time.time_ns() // 1_000_000
        if self.operator_state is not None:
            self.operator_state.publish_event(
                event_type=event_type,
                occurred_at_unix_milliseconds=occurred_at_unix_milliseconds,
                host_synced_seconds=host_synced_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
                visit_id=visit_id,
                payload=payload,
            )
        if self.world_state is not None and visit_id is not None:
            self.world_state.publish_transition(
                event_type=event_type,
                entity_type="visit",
                entity_id=visit_id,
                payload=payload,
                occurred_at_unix_milliseconds=occurred_at_unix_milliseconds,
                host_synced_seconds=host_synced_seconds,
            )

    def publish_shop_api_entry_result(
        self,
        *,
        event_type: str,
        camera_index: int,
        device_id: str,
        track_id: int,
        visit_id: int | None,
        host_synced_seconds: float,
        payload: dict[str, object],
    ) -> None:
        if event_type not in {
            "shop_entry_bound",
            "shop_entry_bind_skipped",
            "shop_entry_bind_failed",
        }:
            raise ValueError(f"Unsupported shop entry result event: {event_type}")
        occurred_at_unix_milliseconds = time.time_ns() // 1_000_000
        if self.operator_state is not None:
            self.operator_state.publish_event(
                event_type=event_type,
                occurred_at_unix_milliseconds=occurred_at_unix_milliseconds,
                host_synced_seconds=host_synced_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
                visit_id=visit_id,
                payload=payload,
            )
        if self.world_state is not None and visit_id is not None:
            self.world_state.publish_transition(
                event_type=event_type,
                entity_type="visit",
                entity_id=visit_id,
                payload=payload,
                occurred_at_unix_milliseconds=occurred_at_unix_milliseconds,
                host_synced_seconds=host_synced_seconds,
            )

    def publish_customer_binding(
        self,
        *,
        visit_id: int,
        customer_id: str,
        host_synced_seconds: float,
        camera_index: int,
        device_id: str,
        track_id: int,
    ) -> None:
        if self.operator_state is None and self.world_state is None:
            return
        if self.operator_state is not None:
            self.operator_state.publish_visit_state(
                visit_id,
                customer_id=customer_id,
                host_synced_seconds=host_synced_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
            )
            self.operator_state.publish_event(
                event_type="customer_binding_changed",
                host_synced_seconds=host_synced_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
                visit_id=visit_id,
                payload={"customerId": customer_id},
            )
        if self.world_state is not None:
            self.world_state.publish_visit_state(
                visit_id,
                customer_id=customer_id,
                host_synced_seconds=host_synced_seconds,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track_id,
                event_type="customer_binding_changed",
            )

    def _generate_stream(self, camera_index: int) -> Iterator[bytes]:
        with self._condition:
            camera = self._cameras[camera_index]
            last_sequence = camera.sequence - 1 if camera.jpeg is not None else camera.sequence
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopping or self._cameras[camera_index].sequence > last_sequence,
                    timeout=1.0,
                )
                if self._stopping:
                    return

                camera = self._cameras[camera_index]
                if camera.sequence <= last_sequence or camera.jpeg is None:
                    continue
                jpeg = camera.jpeg
                last_sequence = camera.sequence

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def stop(self, shutdown_timeout_seconds: float = 5.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=shutdown_timeout_seconds)
            if self._thread.is_alive():
                print("Warning: MJPEG streaming API did not stop before the shutdown timeout.")
        if self.operator_store is not None:
            self.operator_store.close()
        if self.world_state_store is not None:
            self.world_state_store.close()
