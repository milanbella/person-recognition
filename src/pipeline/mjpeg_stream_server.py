from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from pipeline.observer_api import ObserverCameraSnapshot, observer_snapshot_payload
from pipeline.operator_api import create_operator_router
from pipeline.operator_state import OperatorState
from pipeline.operator_test_store import OperatorTestStore
from pipeline.shelf_api import (
    ShelfCameraSnapshot,
    shelf_camera_snapshot_payload,
    shelf_status_payload,
)
from pipeline.shelf_proximity import ShelfProximityStatus
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
        world_state_db: Path | None = None,
        operator_state_db: Path | None = None,
        operator_runs_root: Path = Path("test-runs"),
        operator_api_token: str | None = None,
        operator_runtime_configuration: dict[str, object] | None = None,
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
        self._cameras = {
            index: _CameraFrame(device_id=device_id, camera_role=camera_roles[index])
            for index, device_id in enumerate(camera_device_ids)
        }
        self._condition = threading.Condition()
        self._stopping = False
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._server_error: BaseException | None = None
        self._shelf_statuses: tuple[ShelfProximityStatus, ...] = ()
        self._shelf_event_payloads: dict[int, dict[str, object]] = {}
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
        self.app = self._build_app()
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
                )
            )

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/stream/{cam_index}")
        def stream(cam_index: int) -> StreamingResponse:
            if cam_index not in self._cameras:
                raise HTTPException(status_code=404, detail=f"Camera index {cam_index} is not configured.")
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
                print(f"MJPEG streaming API listening on http://{self.host}:{self.port}")
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
            cameras = [
                {
                    "id": index,
                    "deviceId": camera.device_id,
                    "status": (
                        "active"
                        if camera.last_frame_monotonic is not None
                        and now - camera.last_frame_monotonic <= self.camera_timeout_seconds
                        else "offline"
                    ),
                }
                for index, camera in self._cameras.items()
            ]
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
