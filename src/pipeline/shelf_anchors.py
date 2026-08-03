from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from pipeline.aruco_markers import ArucoMarkerDetection
from pipeline.depth import CameraIntrinsics, pixel_to_camera_point_mm
from pipeline.shelf_config import ShelfDefinition, ShelfWatchingConfig


@dataclass(frozen=True)
class ShelfMarkerDetectionSnapshot:
    shelf_id: int
    marker_id: int
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float]


@dataclass(frozen=True)
class ShelfAnchorObservation:
    shelf_id: int
    marker_id: int
    device_id: str
    host_synced_seconds: float
    observed_at_unix_milliseconds: int
    center_px: tuple[float, float]
    depth_mm: float
    valid_pixel_count: int
    point_3d_mm: tuple[float, float, float]


@dataclass(frozen=True)
class ShelfAnchor:
    shelf_id: int
    marker_id: int
    device_id: str
    point_3d_mm: tuple[float, float, float]
    sample_count: int
    rms_spread_mm: float
    updated_at_unix_milliseconds: int
    source: str


def configured_shelf_marker_detections(
    detections: Sequence[ArucoMarkerDetection],
    shelves: Sequence[ShelfDefinition],
) -> tuple[ShelfMarkerDetectionSnapshot, ...]:
    shelves_by_marker = {
        marker_id: shelf
        for shelf in shelves
        for marker_id in shelf.all_marker_ids
    }
    return tuple(
        ShelfMarkerDetectionSnapshot(
            shelf_id=shelves_by_marker[detection.marker_id].shelf_id,
            marker_id=detection.marker_id,
            corners_px=detection.corners_px,
            center_px=detection.center_px,
        )
        for detection in detections
        if detection.marker_id in shelves_by_marker
    )


def sample_shelf_marker_anchor(
    depth_frame_mm: np.ndarray,
    detection: ShelfMarkerDetectionSnapshot,
    *,
    device_id: str,
    intrinsics: CameraIntrinsics,
    host_synced_seconds: float,
    observed_at_unix_milliseconds: int,
    min_valid_pixels: int = 5,
    polygon_inset_fraction: float = 0.20,
) -> ShelfAnchorObservation | None:
    if depth_frame_mm.ndim != 2:
        raise ValueError("Depth frame must be a single-channel millimeter image.")
    if not 0.0 <= polygon_inset_fraction < 0.5:
        raise ValueError("Marker polygon inset fraction must be between 0.0 and 0.5.")
    if min_valid_pixels <= 0:
        raise ValueError("Marker minimum valid pixels must be positive.")

    frame_height, frame_width = depth_frame_mm.shape[:2]
    corners = np.asarray(detection.corners_px, dtype=np.float32).reshape(-1, 2)
    if corners.shape[0] < 3:
        return None
    center = corners.mean(axis=0)
    inset = center + ((corners - center) * (1.0 - polygon_inset_fraction))
    x1 = max(0, int(math.floor(float(inset[:, 0].min()))))
    y1 = max(0, int(math.floor(float(inset[:, 1].min()))))
    x2 = min(frame_width, int(math.ceil(float(inset[:, 0].max()))) + 1)
    y2 = min(frame_height, int(math.ceil(float(inset[:, 1].max()))) + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    local_polygon = np.rint(inset - np.asarray([x1, y1])).astype(np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_polygon, 1)
    roi = depth_frame_mm[y1:y2, x1:x2]
    valid_mask = (mask > 0) & (roi > 0) & np.isfinite(roi)
    valid = roi[valid_mask]
    if valid.size < min_valid_pixels:
        return None

    center_x = min(frame_width - 1, max(0, int(round(detection.center_px[0]))))
    center_y = min(frame_height - 1, max(0, int(round(detection.center_px[1]))))
    depth_mm = float(np.median(valid))
    return ShelfAnchorObservation(
        shelf_id=detection.shelf_id,
        marker_id=detection.marker_id,
        device_id=device_id,
        host_synced_seconds=host_synced_seconds,
        observed_at_unix_milliseconds=observed_at_unix_milliseconds,
        center_px=detection.center_px,
        depth_mm=depth_mm,
        valid_pixel_count=int(valid.size),
        point_3d_mm=pixel_to_camera_point_mm(
            pixel_x=center_x,
            pixel_y=center_y,
            depth_mm=depth_mm,
            intrinsics=intrinsics,
        ),
    )


def _point_distance_mm(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def stabilized_anchor(
    observations: Sequence[ShelfAnchorObservation],
    *,
    source: str,
) -> ShelfAnchor | None:
    if not observations:
        return None
    first = observations[0]
    if any(
        observation.shelf_id != first.shelf_id
        or observation.marker_id != first.marker_id
        or observation.device_id != first.device_id
        for observation in observations
    ):
        raise ValueError("Shelf anchor observations must describe one shelf and camera.")
    points = np.asarray(
        [observation.point_3d_mm for observation in observations],
        dtype=np.float64,
    )
    point = np.median(points, axis=0)
    distances = np.linalg.norm(points - point, axis=1)
    rms_spread_mm = float(np.sqrt(np.mean(np.square(distances))))
    return ShelfAnchor(
        shelf_id=first.shelf_id,
        marker_id=first.marker_id,
        device_id=first.device_id,
        point_3d_mm=(float(point[0]), float(point[1]), float(point[2])),
        sample_count=len(observations),
        rms_spread_mm=rms_spread_mm,
        updated_at_unix_milliseconds=max(
            observation.observed_at_unix_milliseconds
            for observation in observations
        ),
        source=source,
    )


class ShelfAnchorManager:
    def __init__(
        self,
        *,
        device_id: str,
        config: ShelfWatchingConfig,
        calibration_root: Path,
        min_samples: int = 10,
        max_spread_mm: float = 50.0,
        movement_tolerance_mm: float = 150.0,
        marker_min_valid_pixels: int = 5,
        auto_save: bool = True,
        load_existing: bool = True,
    ) -> None:
        if min_samples <= 0:
            raise ValueError("Shelf anchor minimum samples must be positive.")
        if max_spread_mm <= 0.0:
            raise ValueError("Shelf anchor maximum spread must be positive.")
        if movement_tolerance_mm <= 0.0:
            raise ValueError("Shelf anchor movement tolerance must be positive.")
        self.device_id = device_id
        self.config = config
        self.shelves = config.shelves
        self.calibration_path = calibration_root / f"shelf_anchors_{device_id}.json"
        self.min_samples = min_samples
        self.max_spread_mm = max_spread_mm
        self.movement_tolerance_mm = movement_tolerance_mm
        self.marker_min_valid_pixels = marker_min_valid_pixels
        self.auto_save = auto_save
        marker_keys = tuple(
            (shelf.shelf_id, marker_id)
            for shelf in self.shelves
            for marker_id in shelf.all_marker_ids
        )
        self._anchors: dict[tuple[int, int], ShelfAnchor] = {}
        self._observations: dict[
            tuple[int, int], deque[ShelfAnchorObservation]
        ] = {
            key: deque(maxlen=max(60, min_samples * 3)) for key in marker_keys
        }
        self._accepted_since_update: dict[tuple[int, int], int] = {
            key: 0 for key in marker_keys
        }
        if load_existing:
            self._load()

    @property
    def anchors(self) -> Mapping[tuple[int, int], ShelfAnchor]:
        return dict(self._anchors)

    @property
    def anchors_by_shelf(self) -> Mapping[int, tuple[ShelfAnchor, ...]]:
        return {
            shelf.shelf_id: self.anchors_for_shelf(shelf.shelf_id)
            for shelf in self.shelves
            if self.anchors_for_shelf(shelf.shelf_id)
        }

    @property
    def anchored_shelves(self) -> tuple[ShelfDefinition, ...]:
        return tuple(
            shelf
            for shelf in self.shelves
            if self.anchors_for_shelf(shelf.shelf_id)
        )

    def anchor_for_shelf(self, shelf_id: int) -> ShelfAnchor | None:
        anchors = self.anchors_for_shelf(shelf_id)
        return anchors[0] if anchors else None

    def anchors_for_shelf(self, shelf_id: int) -> tuple[ShelfAnchor, ...]:
        shelf = self.config.shelf_by_id().get(shelf_id)
        if shelf is None:
            return ()
        return tuple(
            anchor
            for marker_id in shelf.all_marker_ids
            if (anchor := self._anchors.get((shelf_id, marker_id))) is not None
        )

    def process_detections(
        self,
        detections: Sequence[ShelfMarkerDetectionSnapshot],
        *,
        depth_frame_mm: np.ndarray,
        intrinsics: CameraIntrinsics,
        host_synced_seconds: float,
        observed_at_unix_milliseconds: int,
        log_trace: bool = False,
    ) -> tuple[ShelfAnchorObservation, ...]:
        accepted: list[ShelfAnchorObservation] = []
        changed = False
        for detection in detections:
            anchor_key = (detection.shelf_id, detection.marker_id)
            if anchor_key not in self._observations:
                continue
            observation = sample_shelf_marker_anchor(
                depth_frame_mm,
                detection,
                device_id=self.device_id,
                intrinsics=intrinsics,
                host_synced_seconds=host_synced_seconds,
                observed_at_unix_milliseconds=observed_at_unix_milliseconds,
                min_valid_pixels=self.marker_min_valid_pixels,
            )
            if observation is None:
                if log_trace:
                    print(
                        f"SHELF_ANCHOR_TRACE shelf_id={detection.shelf_id} "
                        f"marker_id={detection.marker_id} device_id={self.device_id} "
                        "status=invalid_depth"
                    )
                continue

            current = self._anchors.get(anchor_key)
            if (
                current is not None
                and _point_distance_mm(
                    observation.point_3d_mm,
                    current.point_3d_mm,
                )
                > self.movement_tolerance_mm
            ):
                if log_trace:
                    print(
                        f"SHELF_ANCHOR_TRACE shelf_id={observation.shelf_id} "
                        f"marker_id={observation.marker_id} device_id={self.device_id} "
                        "status=rejected_movement "
                        f"delta_mm={_point_distance_mm(observation.point_3d_mm, current.point_3d_mm):.0f}"
                    )
                continue

            accepted.append(observation)
            samples = self._observations[anchor_key]
            samples.append(observation)
            self._accepted_since_update[anchor_key] += 1
            candidate = stabilized_anchor(
                tuple(samples),
                source=(
                    "runtime_auto_calibrated"
                    if current is None
                    else "persisted_and_runtime_validated"
                ),
            )
            if (
                candidate is not None
                and candidate.sample_count >= self.min_samples
                and candidate.rms_spread_mm <= self.max_spread_mm
                and (
                    current is None
                    or self._accepted_since_update[anchor_key] >= self.min_samples
                )
            ):
                self._anchors[anchor_key] = candidate
                self._accepted_since_update[anchor_key] = 0
                changed = True
                if log_trace:
                    print(
                        f"SHELF_ANCHOR_TRACE shelf_id={candidate.shelf_id} "
                        f"marker_id={candidate.marker_id} device_id={self.device_id} "
                        f"status=accepted samples={candidate.sample_count} "
                        f"spread_mm={candidate.rms_spread_mm:.1f}"
                    )
            elif log_trace:
                spread = float("nan") if candidate is None else candidate.rms_spread_mm
                print(
                    f"SHELF_ANCHOR_TRACE shelf_id={observation.shelf_id} "
                    f"marker_id={observation.marker_id} device_id={self.device_id} "
                    f"status=collecting samples={len(samples)} spread_mm={spread:.1f}"
                )
        if changed and self.auto_save:
            self.save()
        return tuple(accepted)

    def candidate_anchor(
        self,
        shelf_id: int,
        marker_id: int | None = None,
    ) -> ShelfAnchor | None:
        shelf = self.config.shelf_by_id().get(shelf_id)
        if shelf is None:
            return None
        selected_marker_id = (
            shelf.all_marker_ids[0] if marker_id is None else marker_id
        )
        samples = self._observations.get((shelf_id, selected_marker_id))
        if not samples:
            return None
        return stabilized_anchor(tuple(samples), source="calibration_candidate")

    def accept_candidates(self) -> None:
        changed = False
        for shelf in self.shelves:
            for marker_id in shelf.all_marker_ids:
                candidate = self.candidate_anchor(shelf.shelf_id, marker_id)
                if candidate is None or candidate.sample_count < self.min_samples:
                    continue
                if candidate.rms_spread_mm > self.max_spread_mm:
                    continue
                self._anchors[(shelf.shelf_id, marker_id)] = ShelfAnchor(
                    **{
                        **candidate.__dict__,
                        "source": "operator_calibrated",
                    }
                )
                changed = True
        if changed:
            self.save()

    def save(self) -> None:
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": 2,
            "deviceId": self.device_id,
            "arucoDictionary": self.config.aruco_dictionary,
            "anchors": [
                {
                    "shelfId": anchor.shelf_id,
                    "markerId": anchor.marker_id,
                    "point3dMm": list(anchor.point_3d_mm),
                    "sampleCount": anchor.sample_count,
                    "rmsSpreadMm": anchor.rms_spread_mm,
                    "updatedAtUnixMilliseconds": anchor.updated_at_unix_milliseconds,
                    "source": anchor.source,
                }
                for anchor in sorted(
                    self._anchors.values(),
                    key=lambda item: (item.shelf_id, item.marker_id),
                )
            ],
        }
        temporary_path = self.calibration_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.calibration_path)

    def _load(self) -> None:
        if not self.calibration_path.exists():
            return
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        if payload.get("schemaVersion") not in {1, 2}:
            raise ValueError(
                f"Unsupported shelf anchor schema in {self.calibration_path}."
            )
        if payload.get("deviceId") != self.device_id:
            raise ValueError(
                f"Shelf anchor device mismatch in {self.calibration_path}."
            )
        if payload.get("arucoDictionary") != self.config.aruco_dictionary:
            raise ValueError(
                f"Shelf anchor dictionary mismatch in {self.calibration_path}."
            )
        shelves_by_marker = {
            marker_id: shelf
            for shelf in self.shelves
            for marker_id in shelf.all_marker_ids
        }
        for raw_anchor in payload.get("anchors", []):
            marker_id = int(raw_anchor["markerId"])
            shelf = shelves_by_marker.get(marker_id)
            if shelf is None:
                continue
            shelf_id = shelf.shelf_id
            point = raw_anchor["point3dMm"]
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(
                    f"Shelf {shelf_id} point3dMm is invalid in {self.calibration_path}."
                )
            anchor_key = (shelf_id, marker_id)
            if anchor_key in self._anchors:
                raise ValueError(
                    f"Duplicate shelf anchor {anchor_key} in {self.calibration_path}."
                )
            self._anchors[anchor_key] = ShelfAnchor(
                shelf_id=shelf_id,
                marker_id=marker_id,
                device_id=self.device_id,
                point_3d_mm=tuple(float(value) for value in point),
                sample_count=int(raw_anchor.get("sampleCount", 0)),
                rms_spread_mm=float(raw_anchor.get("rmsSpreadMm", 0.0)),
                updated_at_unix_milliseconds=int(
                    raw_anchor.get("updatedAtUnixMilliseconds", 0)
                ),
                source=str(raw_anchor.get("source", "persisted")),
            )
