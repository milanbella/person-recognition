from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import depthai as dai


@dataclass(frozen=True)
class ProductTrainingCamera:
    device_id: str
    control_queue: Any
    frame_queue: Any
    context_provider: Callable[[], Mapping[str, object]]


class ProductTrainingCaptureService:
    def __init__(
        self,
        root: Path,
        *,
        jpeg_quality: int = 95,
        timeout_seconds: float = 5.0,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("Training capture JPEG quality must be between 1 and 100.")
        if timeout_seconds <= 0:
            raise ValueError("Training capture timeout must be greater than zero.")
        if (target_width is None) != (target_height is None):
            raise ValueError("Training capture target width and height must be set together.")
        if target_width is not None and (target_width <= 0 or target_height <= 0):
            raise ValueError("Training capture target dimensions must be greater than zero.")
        self.root = root
        self.jpeg_quality = jpeg_quality
        self.timeout_seconds = timeout_seconds
        self.target_width = target_width
        self.target_height = target_height
        self._condition = threading.Lock()
        self._cameras: dict[int, ProductTrainingCamera] = {}
        self._camera_locks: dict[int, threading.Lock] = {}

    def register_camera(
        self,
        camera_index: int,
        *,
        device_id: str,
        control_queue: Any,
        frame_queue: Any,
        context_provider: Callable[[], Mapping[str, object]],
    ) -> None:
        with self._condition:
            self._cameras[camera_index] = ProductTrainingCamera(
                device_id=device_id,
                control_queue=control_queue,
                frame_queue=frame_queue,
                context_provider=context_provider,
            )
            self._camera_locks.setdefault(camera_index, threading.Lock())

    def capture(self, camera_index: int) -> dict[str, object]:
        with self._condition:
            camera = self._cameras.get(camera_index)
            camera_lock = self._camera_locks.get(camera_index)
        if camera is None or camera_lock is None:
            raise KeyError(f"Camera index {camera_index} is not ready for 4K capture.")

        with camera_lock:
            while camera.frame_queue.tryGet() is not None:
                pass
            control = dai.CameraControl()
            control.setCaptureStill(True)
            camera.control_queue.send(control)
            frame_message = camera.frame_queue.get(
                timedelta(seconds=self.timeout_seconds)
            )
            if frame_message is None:
                raise TimeoutError(
                    f"Timed out waiting for camera {camera_index + 1} 4K frame."
                )
            frame = frame_message.getCvFrame()
            if frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Camera {camera_index + 1} returned an empty 4K frame."
                )
            source_height, source_width = frame.shape[:2]
            if self.target_width is not None and self.target_height is not None:
                frame = _center_crop_resize(
                    frame,
                    target_width=self.target_width,
                    target_height=self.target_height,
                )
            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not success:
                raise RuntimeError(
                    f"Could not encode camera {camera_index + 1} training frame."
                )

            captured_at = datetime.now(timezone.utc)
            captured_ms = int(captured_at.timestamp() * 1000)
            sequence_number = int(frame_message.getSequenceNum())
            stem = (
                captured_at.strftime("%Y%m%d-%H%M%S")
                + f"-{captured_ms % 1000:03d}-seq-{sequence_number}"
            )
            camera_dir = self.root / f"camera-{camera_index + 1}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            image_path = camera_dir / f"{stem}.jpg"
            metadata_path = camera_dir / f"{stem}.json"

            context = dict(camera.context_provider())
            metadata: dict[str, object] = {
                "cameraNumber": camera_index + 1,
                "cameraIndex": camera_index,
                "deviceId": camera.device_id,
                "capturedAt": captured_at.isoformat().replace("+00:00", "Z"),
                "capturedAtUnixMilliseconds": captured_ms,
                "rgbSequenceNumber": sequence_number,
                "hostSyncedSeconds": float(
                    frame_message.getTimestamp().total_seconds()
                ),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "sourceWidth": int(source_width),
                "sourceHeight": int(source_height),
                "resizeMode": "center_crop",
                "jpegQuality": self.jpeg_quality,
                "imageFile": image_path.name,
                "context": context,
            }
            image_path.write_bytes(encoded.tobytes())
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "cameraNumber": camera_index + 1,
                "cameraIndex": camera_index,
                "deviceId": camera.device_id,
                "capturedAtUnixMilliseconds": captured_ms,
                "rgbSequenceNumber": sequence_number,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "sourceWidth": int(source_width),
                "sourceHeight": int(source_height),
                "imagePath": str(image_path),
                "metadataPath": str(metadata_path),
            }


def _center_crop_resize(
    frame: Any,
    *,
    target_width: int,
    target_height: int,
) -> Any:
    source_height, source_width = frame.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = source_width / source_height
    if source_aspect > target_aspect:
        crop_width = max(1, round(source_height * target_aspect))
        x1 = (source_width - crop_width) // 2
        cropped = frame[:, x1 : x1 + crop_width]
    else:
        crop_height = max(1, round(source_width / target_aspect))
        y1 = (source_height - crop_height) // 2
        cropped = frame[y1 : y1 + crop_height, :]
    if cropped.shape[1] == target_width and cropped.shape[0] == target_height:
        return cropped
    return cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
