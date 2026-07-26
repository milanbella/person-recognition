from __future__ import annotations

from typing import Iterable

import numpy as np

from pipeline.depth import CameraIntrinsics, DepthSample, pixel_to_camera_point_mm
from pipeline.shelf_config import ShelfPersonDepthConfig
from pipeline.tracking import Track


def _roi_for_track(
    track: Track,
    *,
    frame_width: int,
    frame_height: int,
    center_x_fraction: float,
    center_y_fraction: float,
    width_fraction: float,
    height_fraction: float,
) -> tuple[int, int, int, int]:
    box_width = max(1, track.x2 - track.x1)
    box_height = max(1, track.y2 - track.y1)
    roi_width = max(6, int(round(box_width * width_fraction)))
    roi_height = max(6, int(round(box_height * height_fraction)))
    center_x = track.x1 + (box_width * center_x_fraction)
    center_y = track.y1 + (box_height * center_y_fraction)

    x1 = max(0, int(round(center_x - (roi_width / 2.0))))
    y1 = max(0, int(round(center_y - (roi_height / 2.0))))
    x2 = min(frame_width, x1 + roi_width)
    y2 = min(frame_height, y1 + roi_height)
    if x2 - x1 < roi_width:
        x1 = max(0, x2 - roi_width)
    if y2 - y1 < roi_height:
        y1 = max(0, y2 - roi_height)
    return x1, y1, x2, y2


def shelf_person_depth_rois(
    track: Track,
    *,
    frame_width: int,
    frame_height: int,
    config: ShelfPersonDepthConfig,
) -> tuple[tuple[int, int, int, int], ...]:
    center_y_values: Iterable[float] = (
        config.center_y_fraction,
        *config.fallback_center_y_fractions,
    )
    rois: list[tuple[int, int, int, int]] = []
    for center_y_fraction in center_y_values:
        roi = _roi_for_track(
            track,
            frame_width=frame_width,
            frame_height=frame_height,
            center_x_fraction=config.center_x_fraction,
            center_y_fraction=center_y_fraction,
            width_fraction=config.width_fraction,
            height_fraction=config.height_fraction,
        )
        if roi not in rois:
            rois.append(roi)
    return tuple(rois)


def sample_shelf_person_depth(
    depth_frame_mm: np.ndarray,
    track: Track,
    *,
    intrinsics: CameraIntrinsics,
    config: ShelfPersonDepthConfig,
) -> DepthSample | None:
    if depth_frame_mm.ndim != 2:
        raise ValueError("Depth frame must be a single-channel millimeter image.")
    frame_height, frame_width = depth_frame_mm.shape[:2]
    for x1, y1, x2, y2 in shelf_person_depth_rois(
        track,
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
    ):
        if x2 <= x1 or y2 <= y1:
            continue
        roi = depth_frame_mm[y1:y2, x1:x2]
        valid = roi[(roi > 0) & np.isfinite(roi)]
        if valid.size < config.min_valid_pixels:
            continue
        anchor_px = (
            int(round((x1 + x2) / 2.0)),
            int(round((y1 + y2) / 2.0)),
        )
        depth_mm = float(np.median(valid))
        return DepthSample(
            depth_mm=depth_mm,
            valid_pixel_count=int(valid.size),
            roi=(x1, y1, x2, y2),
            anchor_px=anchor_px,
            point_3d_mm=pixel_to_camera_point_mm(
                pixel_x=anchor_px[0],
                pixel_y=anchor_px[1],
                depth_mm=depth_mm,
                intrinsics=intrinsics,
            ),
        )
    return None
