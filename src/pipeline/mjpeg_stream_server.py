from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from pipeline.observer_api import ObserverCameraSnapshot, observer_snapshot_payload
from pipeline.visit_registry import is_observer_enabled


@dataclass
class _CameraFrame:
    device_id: str
    camera_role: str
    jpeg: bytes | None = None
    sequence: int = 0
    last_frame_monotonic: float | None = None
    observer_snapshot: ObserverCameraSnapshot | None = None
    observer_snapshot_monotonic: float | None = None


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
        self.app = self._build_app()

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

    def publish(self, camera_index: int, frame: np.ndarray) -> None:
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
            self._condition.notify_all()

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
