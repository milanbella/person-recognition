from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from model_training.store import ModelTrainingStore


class CaptureRegistrar:
    def __init__(self, root: Path, store: ModelTrainingStore) -> None:
        self.root = Path(root).resolve()
        self.store = store
        self.captures_root = self.root / "captures" / "sessions"
        self.proxies_root = self.root / "captures" / "review-proxies"
        self.thumbnails_root = self.root / "captures" / "thumbnails"
        for directory in (self.captures_root, self.proxies_root, self.thumbnails_root):
            directory.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        session: Mapping[str, Any],
        capture: Mapping[str, Any],
        *,
        capture_request_id: str,
    ) -> dict[str, Any]:
        source_image = Path(str(capture["imagePath"])).resolve(strict=True)
        source_metadata = Path(str(capture["metadataPath"])).resolve(strict=True)
        if not source_image.is_file() or not source_metadata.is_file():
            raise RuntimeError("Live capture response does not reference regular files.")

        frame_id = str(uuid.uuid4())
        session_id = str(session["sessionId"])
        session_dir = self.captures_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        image_path = session_dir / f"{frame_id}.jpg"
        metadata_path = session_dir / f"{frame_id}.json"
        proxy_path = self.proxies_root / f"{frame_id}.jpg"
        thumbnail_path = self.thumbnails_root / f"{frame_id}.jpg"
        created: list[Path] = []
        try:
            _link_or_copy(source_image, image_path)
            created.append(image_path)
            _link_or_copy(source_metadata, metadata_path)
            created.append(metadata_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise RuntimeError(f"Could not decode captured image: {image_path}")
            height, width = image.shape[:2]
            _write_resized(image, proxy_path, 1280, 720, quality=90)
            created.append(proxy_path)
            _write_resized(image, thumbnail_path, 320, 180, quality=82)
            created.append(thumbnail_path)
            return self.store.add_frame(
                {
                    "frame_id": frame_id,
                    "capture_request_id": capture_request_id,
                    "session_id": session_id,
                    "camera_index": int(capture["cameraIndex"]),
                    "device_id": str(capture["deviceId"]),
                    "rgb_sequence_number": capture.get("rgbSequenceNumber"),
                    "captured_at_unix_ms": int(capture["capturedAtUnixMilliseconds"]),
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "source_capture_path": str(source_image),
                    "review_proxy_path": str(proxy_path),
                    "thumbnail_path": str(thumbnail_path),
                    "width": width,
                    "height": height,
                    "sha256": _sha256(image_path),
                    "perceptual_hash": _perceptual_hash(image),
                }
            )
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        if _sha256(source) != _sha256(temporary):
            raise RuntimeError(f"Copied file hash mismatch: {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_resized(image: np.ndarray, path: Path, max_width: int, max_height: int, *, quality: int) -> None:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    resized = image if scale == 1.0 else cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode image derivative: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    values = coefficients.flatten()[1:]
    median = float(np.median(values))
    bits = "".join("1" if value >= median else "0" for value in coefficients.flatten())
    return f"{int(bits, 2):016x}"
