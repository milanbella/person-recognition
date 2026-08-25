from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


LEGACY_SHELF_REGION_SCHEMA_VERSION = 1
SHELF_REGION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class ShelfDepthCell:
    column: int
    row: int
    median_depth_mm: float
    mad_depth_mm: float
    valid_sample_count: int
    valid_fraction: float


@dataclass(frozen=True)
class ShelfDepthModel:
    columns: int
    rows: int
    aligned_depth_width: int
    aligned_depth_height: int
    minimum_valid_fraction: float
    cells: tuple[ShelfDepthCell, ...]
    calibration_frame_count: int
    calibrated_at_unix_milliseconds: int
    intrinsics_fx: float
    intrinsics_fy: float
    intrinsics_cx: float
    intrinsics_cy: float
    camera_calibration_id: str

    def by_grid_position(self) -> dict[tuple[int, int], ShelfDepthCell]:
        return {(cell.column, cell.row): cell for cell in self.cells}


@dataclass(frozen=True)
class ShelfRegion:
    shelf_id: int
    polygon: tuple[NormalizedPoint, ...]
    depth_tolerance_mm: float | None = None
    depth_model: ShelfDepthModel | None = None
    off_shelf_hysteresis_mm: float = 150.0
    additional_polygons: tuple[tuple[NormalizedPoint, ...], ...] = ()

    @property
    def polygons(self) -> tuple[tuple[NormalizedPoint, ...], ...]:
        return (self.polygon, *self.additional_polygons)


@dataclass(frozen=True)
class CameraShelfRegions:
    device_id: str
    regions: tuple[ShelfRegion, ...]
    schema_version: int = SHELF_REGION_SCHEMA_VERSION

    def by_shelf_id(self) -> dict[int, ShelfRegion]:
        return {region.shelf_id: region for region in self.regions}

    @property
    def has_3d_calibration(self) -> bool:
        return (
            self.schema_version == SHELF_REGION_SCHEMA_VERSION
            and bool(self.regions)
            and all(region.depth_model is not None for region in self.regions)
        )


def shelf_regions_path(root: Path, device_id: str) -> Path:
    return root / f"shelf_regions_{device_id}.json"


def depth_calibration_identity(
    device_id: str,
    *,
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
) -> str:
    canonical = json.dumps(
        {
            "deviceId": device_id,
            "width": width,
            "height": height,
            "intrinsics": [round(value, 6) for value in intrinsics],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_point(payload: Mapping[str, Any], *, field_name: str) -> NormalizedPoint:
    unknown = set(payload) - {"x", "y"}
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {sorted(unknown)}")
    try:
        x = float(payload["x"])
        y = float(payload["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} requires numeric x and y values.") from exc
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError(f"{field_name} coordinates must be between zero and one.")
    return NormalizedPoint(x, y)


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _depth_model_from_payload(
    payload: Mapping[str, Any], *, field_name: str
) -> ShelfDepthModel:
    expected = {
        "type", "columns", "rows", "alignedDepthWidth", "alignedDepthHeight",
        "minimumValidFraction", "cells", "calibrationFrameCount",
        "calibratedAtUnixMilliseconds", "intrinsics", "cameraCalibrationId",
    }
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {sorted(unknown)}")
    if payload.get("type") != "normalized_depth_grid":
        raise ValueError(f"{field_name}.type must be normalized_depth_grid.")
    columns = _non_negative_int(payload.get("columns"), field_name=f"{field_name}.columns")
    rows = _non_negative_int(payload.get("rows"), field_name=f"{field_name}.rows")
    width = _non_negative_int(payload.get("alignedDepthWidth"), field_name=f"{field_name}.alignedDepthWidth")
    height = _non_negative_int(payload.get("alignedDepthHeight"), field_name=f"{field_name}.alignedDepthHeight")
    if min(columns, rows, width, height) <= 0:
        raise ValueError(f"{field_name} grid dimensions must be positive.")
    minimum_valid_fraction = _positive_float(
        payload.get("minimumValidFraction"), field_name=f"{field_name}.minimumValidFraction"
    )
    if minimum_valid_fraction > 1.0:
        raise ValueError(f"{field_name}.minimumValidFraction must not exceed one.")
    frame_count = _non_negative_int(payload.get("calibrationFrameCount"), field_name=f"{field_name}.calibrationFrameCount")
    calibrated_at = _non_negative_int(payload.get("calibratedAtUnixMilliseconds"), field_name=f"{field_name}.calibratedAtUnixMilliseconds")
    if frame_count <= 0:
        raise ValueError(f"{field_name}.calibrationFrameCount must be positive.")
    intrinsics = payload.get("intrinsics")
    if not isinstance(intrinsics, dict) or set(intrinsics) != {"fx", "fy", "cx", "cy"}:
        raise ValueError(f"{field_name}.intrinsics requires fx, fy, cx, and cy.")
    calibration_id = payload.get("cameraCalibrationId")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError(f"{field_name}.cameraCalibrationId must be non-empty.")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError(f"{field_name}.cells must be a list.")
    cells: list[ShelfDepthCell] = []
    seen: set[tuple[int, int]] = set()
    expected_cell_fields = {
        "column", "row", "medianDepthMm", "madDepthMm",
        "validSampleCount", "validFraction",
    }
    for index, raw_cell in enumerate(raw_cells):
        cell_name = f"{field_name}.cells[{index}]"
        if not isinstance(raw_cell, dict) or set(raw_cell) != expected_cell_fields:
            raise ValueError(f"{cell_name} has invalid fields.")
        column = _non_negative_int(raw_cell["column"], field_name=f"{cell_name}.column")
        row = _non_negative_int(raw_cell["row"], field_name=f"{cell_name}.row")
        if column >= columns or row >= rows:
            raise ValueError(f"{cell_name} lies outside the depth grid.")
        if (column, row) in seen:
            raise ValueError(f"Duplicate shelf depth cell {(column, row)}.")
        seen.add((column, row))
        valid_fraction = _positive_float(raw_cell["validFraction"], field_name=f"{cell_name}.validFraction")
        if valid_fraction > 1.0:
            raise ValueError(f"{cell_name}.validFraction must not exceed one.")
        mad = float(raw_cell["madDepthMm"])
        if not np.isfinite(mad) or mad < 0.0:
            raise ValueError(f"{cell_name}.madDepthMm must be non-negative.")
        cells.append(ShelfDepthCell(
            column=column,
            row=row,
            median_depth_mm=_positive_float(raw_cell["medianDepthMm"], field_name=f"{cell_name}.medianDepthMm"),
            mad_depth_mm=mad,
            valid_sample_count=_non_negative_int(raw_cell["validSampleCount"], field_name=f"{cell_name}.validSampleCount"),
            valid_fraction=valid_fraction,
        ))
    if not cells:
        raise ValueError(f"{field_name}.cells must contain valid calibrated cells.")
    return ShelfDepthModel(
        columns=columns, rows=rows,
        aligned_depth_width=width, aligned_depth_height=height,
        minimum_valid_fraction=minimum_valid_fraction, cells=tuple(cells),
        calibration_frame_count=frame_count,
        calibrated_at_unix_milliseconds=calibrated_at,
        intrinsics_fx=_positive_float(intrinsics["fx"], field_name=f"{field_name}.intrinsics.fx"),
        intrinsics_fy=_positive_float(intrinsics["fy"], field_name=f"{field_name}.intrinsics.fy"),
        intrinsics_cx=float(intrinsics["cx"]), intrinsics_cy=float(intrinsics["cy"]),
        camera_calibration_id=calibration_id,
    )


def camera_shelf_regions_from_payload(payload: Mapping[str, Any]) -> CameraShelfRegions:
    unknown = set(payload) - {"schemaVersion", "deviceId", "shelves"}
    if unknown:
        raise ValueError(f"Shelf-region config contains unknown fields: {sorted(unknown)}")
    schema_version = payload.get("schemaVersion")
    if schema_version not in {LEGACY_SHELF_REGION_SCHEMA_VERSION, SHELF_REGION_SCHEMA_VERSION}:
        raise ValueError("Shelf-region schemaVersion must be 1 (diagnostic only) or 2.")
    device_id = payload.get("deviceId")
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("Shelf-region deviceId must be a non-empty string.")
    shelves = payload.get("shelves")
    if not isinstance(shelves, list):
        raise ValueError("Shelf-region shelves must be a list.")
    regions: list[ShelfRegion] = []
    seen: set[int] = set()
    for index, raw_region in enumerate(shelves):
        if not isinstance(raw_region, dict):
            raise ValueError(f"shelves[{index}] must be an object.")
        allowed = {"shelfId", "polygonNormalized", "polygonsNormalized", "depthToleranceMm", "onShelfToleranceMm", "offShelfHysteresisMm", "depthModel"}
        unknown_region = set(raw_region) - allowed
        if unknown_region:
            raise ValueError(f"shelves[{index}] contains unknown fields: {sorted(unknown_region)}")
        shelf_id = _non_negative_int(raw_region.get("shelfId"), field_name=f"shelves[{index}].shelfId")
        if shelf_id in seen:
            raise ValueError(f"Duplicate shelfId in shelf-region config: {shelf_id}")
        seen.add(shelf_id)
        has_singular = "polygonNormalized" in raw_region
        has_plural = "polygonsNormalized" in raw_region
        if has_singular == has_plural:
            raise ValueError(
                f"shelves[{index}] requires exactly one of polygonNormalized "
                "or polygonsNormalized."
            )
        raw_polygons = (
            [raw_region["polygonNormalized"]]
            if has_singular
            else raw_region["polygonsNormalized"]
        )
        if not isinstance(raw_polygons, list) or not raw_polygons:
            raise ValueError(f"shelves[{index}].polygonsNormalized must be non-empty.")
        polygons: list[tuple[NormalizedPoint, ...]] = []
        for polygon_index, raw_polygon in enumerate(raw_polygons):
            field_name = f"shelves[{index}].polygonsNormalized[{polygon_index}]"
            if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
                raise ValueError(f"{field_name} requires at least three points.")
            if not all(isinstance(point, dict) for point in raw_polygon):
                raise ValueError(f"{field_name} points must be objects.")
            polygons.append(
                tuple(
                    _normalized_point(
                        point,
                        field_name=f"{field_name}[{point_index}]",
                    )
                    for point_index, point in enumerate(raw_polygon)
                )
            )
        tolerance_value = raw_region.get("onShelfToleranceMm", raw_region.get("depthToleranceMm"))
        tolerance = None if tolerance_value is None else _positive_float(tolerance_value, field_name=f"shelves[{index}].onShelfToleranceMm")
        hysteresis = _positive_float(raw_region.get("offShelfHysteresisMm", 150.0), field_name=f"shelves[{index}].offShelfHysteresisMm")
        raw_depth_model = raw_region.get("depthModel")
        depth_model = None
        if raw_depth_model is not None:
            if not isinstance(raw_depth_model, dict):
                raise ValueError(f"shelves[{index}].depthModel must be an object.")
            depth_model = _depth_model_from_payload(raw_depth_model, field_name=f"shelves[{index}].depthModel")
        if schema_version == SHELF_REGION_SCHEMA_VERSION and depth_model is None:
            raise ValueError(f"shelves[{index}] requires depthModel for schemaVersion 2.")
        regions.append(
            ShelfRegion(
                shelf_id=shelf_id,
                polygon=polygons[0],
                depth_tolerance_mm=tolerance,
                depth_model=depth_model,
                off_shelf_hysteresis_mm=hysteresis,
                additional_polygons=tuple(polygons[1:]),
            )
        )
    return CameraShelfRegions(device_id, tuple(regions), int(schema_version))


def _depth_model_payload(model: ShelfDepthModel) -> dict[str, object]:
    return {
        "type": "normalized_depth_grid", "columns": model.columns, "rows": model.rows,
        "alignedDepthWidth": model.aligned_depth_width,
        "alignedDepthHeight": model.aligned_depth_height,
        "minimumValidFraction": model.minimum_valid_fraction,
        "calibrationFrameCount": model.calibration_frame_count,
        "calibratedAtUnixMilliseconds": model.calibrated_at_unix_milliseconds,
        "intrinsics": {"fx": model.intrinsics_fx, "fy": model.intrinsics_fy, "cx": model.intrinsics_cx, "cy": model.intrinsics_cy},
        "cameraCalibrationId": model.camera_calibration_id,
        "cells": [{
            "column": cell.column, "row": cell.row,
            "medianDepthMm": cell.median_depth_mm, "madDepthMm": cell.mad_depth_mm,
            "validSampleCount": cell.valid_sample_count, "validFraction": cell.valid_fraction,
        } for cell in model.cells],
    }


def camera_shelf_regions_payload(config: CameraShelfRegions) -> dict[str, object]:
    return {
        "schemaVersion": config.schema_version, "deviceId": config.device_id,
        "shelves": [{
            "shelfId": region.shelf_id,
            **(
                {
                    "polygonNormalized": [
                        {"x": point.x, "y": point.y} for point in region.polygon
                    ]
                }
                if config.schema_version == LEGACY_SHELF_REGION_SCHEMA_VERSION
                else {
                    "polygonsNormalized": [
                        [{"x": point.x, "y": point.y} for point in polygon]
                        for polygon in region.polygons
                    ]
                }
            ),
            **({} if region.depth_tolerance_mm is None else {"onShelfToleranceMm": region.depth_tolerance_mm}),
            "offShelfHysteresisMm": region.off_shelf_hysteresis_mm,
            **({} if region.depth_model is None else {"depthModel": _depth_model_payload(region.depth_model)}),
        } for region in config.regions],
    }


def load_camera_shelf_regions(path: Path) -> CameraShelfRegions:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Shelf-region calibration root must be an object.")
    return camera_shelf_regions_from_payload(payload)


def save_camera_shelf_regions(path: Path, config: CameraShelfRegions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(camera_shelf_regions_payload(config), indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def normalized_point_in_polygon(x: float, y: float, polygon: Sequence[NormalizedPoint]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        crosses = (current.y > y) != (previous.y > y)
        if crosses:
            intersection_x = (previous.x - current.x) * (y - current.y) / (previous.y - current.y) + current.x
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def box_center_in_shelf_region(box: tuple[int, int, int, int], *, frame_width: int, frame_height: int, region: ShelfRegion) -> bool:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    center_x = (box[0] + box[2]) / 2.0 / frame_width
    center_y = (box[1] + box[3]) / 2.0 / frame_height
    return any(
        normalized_point_in_polygon(center_x, center_y, polygon)
        for polygon in region.polygons
    )


def shelf_depth_cell_at(region: ShelfRegion, *, normalized_x: float, normalized_y: float) -> ShelfDepthCell | None:
    model = region.depth_model
    if model is None or not any(
        normalized_point_in_polygon(normalized_x, normalized_y, polygon)
        for polygon in region.polygons
    ):
        return None
    column = min(model.columns - 1, max(0, int(normalized_x * model.columns)))
    row = min(model.rows - 1, max(0, int(normalized_y * model.rows)))
    return model.by_grid_position().get((column, row))


def build_shelf_depth_model(
    depth_frames_mm: Sequence[np.ndarray], *, polygons: Sequence[Sequence[NormalizedPoint]],
    columns: int, rows: int, minimum_valid_fraction: float,
    minimum_valid_samples: int, minimum_depth_mm: int, maximum_depth_mm: int,
    calibrated_at_unix_milliseconds: int,
    intrinsics: tuple[float, float, float, float], camera_calibration_id: str,
) -> ShelfDepthModel:
    if not depth_frames_mm:
        raise ValueError("At least one depth frame is required.")
    if columns <= 0 or rows <= 0:
        raise ValueError("Depth-grid dimensions must be positive.")
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be between zero and one.")
    first_shape = depth_frames_mm[0].shape
    if len(first_shape) != 2 or any(frame.shape != first_shape for frame in depth_frames_mm):
        raise ValueError("All depth frames must have one identical two-dimensional shape.")
    height, width = first_shape
    if not polygons:
        raise ValueError("At least one shelf surface polygon is required.")
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_pixels = [
        np.asarray(
            [
                (
                    min(width - 1, max(0, int(round(point.x * (width - 1))))),
                    min(height - 1, max(0, int(round(point.y * (height - 1))))),
                )
                for point in polygon
            ],
            dtype=np.int32,
        )
        for polygon in polygons
    ]
    cv2.fillPoly(mask, polygon_pixels, 1)
    cells: list[ShelfDepthCell] = []
    for row in range(rows):
        y1, y2 = int(round(row * height / rows)), int(round((row + 1) * height / rows))
        for column in range(columns):
            x1, x2 = int(round(column * width / columns)), int(round((column + 1) * width / columns))
            cell_mask = mask[y1:y2, x1:x2].astype(bool)
            possible_per_frame = int(np.count_nonzero(cell_mask))
            if possible_per_frame <= 0:
                continue
            samples = []
            for frame in depth_frames_mm:
                values = frame[y1:y2, x1:x2][cell_mask]
                valid = values[(values >= minimum_depth_mm) & (values <= maximum_depth_mm) & np.isfinite(values)]
                if valid.size:
                    samples.append(valid.astype(np.float32, copy=False))
            if not samples:
                continue
            values = np.concatenate(samples)
            valid_fraction = values.size / (possible_per_frame * len(depth_frames_mm))
            if values.size < minimum_valid_samples or valid_fraction < minimum_valid_fraction:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            cells.append(ShelfDepthCell(column, row, median, mad, int(values.size), float(valid_fraction)))
    if not cells:
        raise ValueError("Shelf polygon produced no valid depth-grid cells.")
    return ShelfDepthModel(
        columns, rows, width, height, minimum_valid_fraction, tuple(cells),
        len(depth_frames_mm), calibrated_at_unix_milliseconds,
        float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3]),
        camera_calibration_id,
    )
