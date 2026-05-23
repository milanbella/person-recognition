from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from pipeline.detection import add_detection_args
from pipeline.tracking import Track, add_tracking_args


DEFAULT_DEPTH_THRESHOLD_MM = 2000
DEFAULT_DEPTH_HYSTERESIS_MM = 250
DEFAULT_PLANE_HYSTERESIS_MM = 150
DEFAULT_DEPTH_MIN_VALID_PIXELS = 25
DEFAULT_DEPTH_ROI_WIDTH_FRACTION = 0.30
DEFAULT_DEPTH_ROI_HEIGHT_FRACTION = 0.22


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Plane3D:
    point_mm: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass
class DepthSample:
    depth_mm: float
    valid_pixel_count: int
    roi: tuple[int, int, int, int]
    anchor_px: tuple[int, int]
    point_3d_mm: tuple[float, float, float]


@dataclass
class DepthEntranceState:
    last_depth_mm: Optional[float] = None
    entered: bool = False
    recent_depths_mm: list[float] = field(default_factory=list)


@dataclass
class DepthEntranceResult:
    entered_track_ids: list[int]
    depth_samples: Dict[int, DepthSample]
    signed_distances_mm: Dict[int, float] = field(default_factory=dict)


def build_depth_entrance_argparser(
    description: str = "Depth-based entrance prototype using stereo depth aligned to RGB.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_detection_args(parser)
    add_tracking_args(parser)
    add_depth_entrance_args(parser)
    return parser


def add_depth_entrance_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plane-json",
        type=Path,
        default=None,
        help=(
            "Optional plane-fit JSON produced by fit_plane_from_tags.py. "
            "When set, plane point/normal and enter direction are loaded from that file."
        ),
    )
    parser.add_argument(
        "--depth-trigger-mode",
        choices=["threshold", "plane"],
        default="threshold",
        help="Use simple depth thresholding or 3D plane crossing for entry detection.",
    )
    parser.add_argument(
        "--depth-threshold-mm",
        type=int,
        default=DEFAULT_DEPTH_THRESHOLD_MM,
        help="Entry threshold in millimeters. Event fires when tracked depth crosses below this value.",
    )
    parser.add_argument(
        "--depth-hysteresis-mm",
        type=int,
        default=DEFAULT_DEPTH_HYSTERESIS_MM,
        help="Required margin above the threshold before a later crossing may trigger an event.",
    )
    parser.add_argument(
        "--plane-point-x-mm",
        type=float,
        default=0.0,
        help="Door plane anchor point X in camera coordinates, millimeters.",
    )
    parser.add_argument(
        "--plane-point-y-mm",
        type=float,
        default=0.0,
        help="Door plane anchor point Y in camera coordinates, millimeters.",
    )
    parser.add_argument(
        "--plane-point-z-mm",
        type=float,
        default=2000.0,
        help="Door plane anchor point Z in camera coordinates, millimeters.",
    )
    parser.add_argument(
        "--plane-normal-x",
        type=float,
        default=0.0,
        help="Door plane normal X component in camera coordinates.",
    )
    parser.add_argument(
        "--plane-normal-y",
        type=float,
        default=0.0,
        help="Door plane normal Y component in camera coordinates.",
    )
    parser.add_argument(
        "--plane-normal-z",
        type=float,
        default=1.0,
        help="Door plane normal Z component in camera coordinates.",
    )
    parser.add_argument(
        "--plane-enter-direction",
        choices=["positive_to_negative", "negative_to_positive"],
        default="positive_to_negative",
        help="Which signed-distance transition counts as entering across the plane.",
    )
    parser.add_argument(
        "--plane-hysteresis-mm",
        type=float,
        default=DEFAULT_PLANE_HYSTERESIS_MM,
        help="Signed-distance hysteresis for plane-crossing rearm logic.",
    )
    parser.add_argument(
        "--depth-min-valid-pixels",
        type=int,
        default=DEFAULT_DEPTH_MIN_VALID_PIXELS,
        help="Minimum number of valid depth pixels required inside the sampling ROI.",
    )
    parser.add_argument(
        "--depth-roi-width-fraction",
        type=float,
        default=DEFAULT_DEPTH_ROI_WIDTH_FRACTION,
        help="Sampling ROI width as a fraction of tracked box width.",
    )
    parser.add_argument(
        "--depth-roi-height-fraction",
        type=float,
        default=DEFAULT_DEPTH_ROI_HEIGHT_FRACTION,
        help="Sampling ROI height as a fraction of tracked box height, anchored to the box bottom.",
    )
    parser.add_argument(
        "--show-depth-window",
        action="store_true",
        help="Show a separate aligned depth window.",
    )
    return parser


def normalize_vector(x: float, y: float, z: float) -> tuple[float, float, float]:
    norm = math.sqrt((x * x) + (y * y) + (z * z))
    if norm <= 1e-12:
        raise ValueError("Plane normal must be non-zero.")
    return (x / norm, y / norm, z / norm)


def load_plane_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "plane_point_mm" not in payload or "plane_normal" not in payload:
        raise ValueError(
            f"Plane JSON must contain 'plane_point_mm' and 'plane_normal': {path}"
        )
    return payload


def resolve_plane_json_path(
    *,
    plane_json: Path | None,
    device_id: str | None = None,
    calibrations_root: Path | None = None,
    recording_dir: Path | None = None,
) -> Path | None:
    if plane_json is not None:
        return plane_json
    if device_id is not None and calibrations_root is not None:
        candidate = calibrations_root / f"plane_fit_{device_id}.json"
        if candidate.exists():
            return candidate
    if recording_dir is None:
        return None
    candidate = recording_dir / "plane_fit.json"
    return candidate if candidate.exists() else None


def plane_from_args(args: argparse.Namespace) -> Plane3D:
    plane_json = getattr(args, "plane_json", None)
    if plane_json is not None:
        payload = load_plane_json(plane_json)
        point_mm = payload["plane_point_mm"]
        normal = payload["plane_normal"]
        return Plane3D(
            point_mm=(
                float(point_mm[0]),
                float(point_mm[1]),
                float(point_mm[2]),
            ),
            normal=normalize_vector(
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
            ),
        )

    return Plane3D(
        point_mm=(
            float(args.plane_point_x_mm),
            float(args.plane_point_y_mm),
            float(args.plane_point_z_mm),
        ),
        normal=normalize_vector(
            float(args.plane_normal_x),
            float(args.plane_normal_y),
            float(args.plane_normal_z),
        ),
    )


def plane_enter_direction_from_args(args: argparse.Namespace) -> str:
    plane_json = getattr(args, "plane_json", None)
    if plane_json is None:
        return str(args.plane_enter_direction)

    payload = load_plane_json(plane_json)
    value = payload.get("recommended_enter_direction_if_person_moves_toward_camera")
    if isinstance(value, str) and value in {"positive_to_negative", "negative_to_positive"}:
        return value
    return str(args.plane_enter_direction)


def intrinsics_from_matrix(matrix: Sequence[Sequence[float]]) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(matrix[0][0]),
        fy=float(matrix[1][1]),
        cx=float(matrix[0][2]),
        cy=float(matrix[1][2]),
    )


def pixel_to_camera_point_mm(
    *,
    pixel_x: int,
    pixel_y: int,
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    z_mm = float(depth_mm)
    x_mm = ((float(pixel_x) - intrinsics.cx) * z_mm) / intrinsics.fx
    y_mm = ((float(pixel_y) - intrinsics.cy) * z_mm) / intrinsics.fy
    return (x_mm, y_mm, z_mm)


def sample_track_depth(
    depth_frame_mm: np.ndarray,
    track: Track,
    *,
    intrinsics: CameraIntrinsics,
    roi_width_fraction: float,
    roi_height_fraction: float,
    min_valid_pixels: int,
) -> DepthSample | None:
    if depth_frame_mm.ndim != 2:
        raise ValueError("Depth frame must be a single-channel millimeter image.")

    frame_height, frame_width = depth_frame_mm.shape[:2]
    box_width = max(1, track.x2 - track.x1)
    box_height = max(1, track.y2 - track.y1)

    roi_width = max(6, int(round(box_width * roi_width_fraction)))
    roi_height = max(6, int(round(box_height * roi_height_fraction)))
    center_x = int(round((track.x1 + track.x2) / 2.0))
    bottom_y = int(round(track.y2))

    x1 = max(0, center_x - (roi_width // 2))
    x2 = min(frame_width, x1 + roi_width)
    y2 = min(frame_height, bottom_y)
    y1 = max(0, y2 - roi_height)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_frame_mm[y1:y2, x1:x2]
    valid = roi[(roi > 0) & np.isfinite(roi)]
    if valid.size < min_valid_pixels:
        return None

    anchor_px = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
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


def process_depth_entrance_logic(
    *,
    tracks: Sequence[Track],
    depth_frame_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    states: Dict[int, DepthEntranceState],
    depth_threshold_mm: float,
    depth_hysteresis_mm: float,
    min_valid_pixels: int,
    roi_width_fraction: float,
    roi_height_fraction: float,
) -> DepthEntranceResult:
    entered_track_ids: list[int] = []
    depth_samples: Dict[int, DepthSample] = {}

    active_ids = {track.track_id for track in tracks if track.status != "REMOVED"}
    for track_id in list(states.keys()):
        if track_id not in active_ids:
            states.pop(track_id, None)

    rearm_depth_mm = depth_threshold_mm + depth_hysteresis_mm

    for track in tracks:
        if track.status not in {"NEW", "TRACKED", "LOST"}:
            continue

        sample = sample_track_depth(
            depth_frame_mm,
            track,
            intrinsics=intrinsics,
            roi_width_fraction=roi_width_fraction,
            roi_height_fraction=roi_height_fraction,
            min_valid_pixels=min_valid_pixels,
        )
        if sample is None:
            continue

        depth_samples[track.track_id] = sample
        state = states.setdefault(track.track_id, DepthEntranceState())
        state.recent_depths_mm.append(sample.depth_mm)
        state.recent_depths_mm = state.recent_depths_mm[-20:]

        if state.last_depth_mm is None:
            state.last_depth_mm = sample.depth_mm
            continue

        crossed = state.last_depth_mm > depth_threshold_mm and sample.depth_mm <= depth_threshold_mm
        if not state.entered and crossed and track.status in {"TRACKED", "LOST"}:
            state.entered = True
            entered_track_ids.append(track.track_id)
        elif state.entered and sample.depth_mm >= rearm_depth_mm:
            state.entered = False

        state.last_depth_mm = sample.depth_mm

    return DepthEntranceResult(
        entered_track_ids=entered_track_ids,
        depth_samples=depth_samples,
    )


def signed_distance_to_plane_mm(point_mm: tuple[float, float, float], plane: Plane3D) -> float:
    px, py, pz = point_mm
    ox, oy, oz = plane.point_mm
    nx, ny, nz = plane.normal
    return ((px - ox) * nx) + ((py - oy) * ny) + ((pz - oz) * nz)


def fit_plane_from_points(points_mm: Sequence[tuple[float, float, float]]) -> Plane3D:
    if len(points_mm) < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    points = np.asarray(points_mm, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize_vector(
        float(vh[-1, 0]),
        float(vh[-1, 1]),
        float(vh[-1, 2]),
    )
    return Plane3D(
        point_mm=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        normal=normal,
    )


def flip_plane_normal(plane: Plane3D) -> Plane3D:
    nx, ny, nz = plane.normal
    return Plane3D(
        point_mm=plane.point_mm,
        normal=(-nx, -ny, -nz),
    )


def orient_plane_normal_toward_positive_z(plane: Plane3D) -> Plane3D:
    if plane.normal[2] >= 0.0:
        return plane
    return flip_plane_normal(plane)


def process_depth_plane_logic(
    *,
    tracks: Sequence[Track],
    depth_frame_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    states: Dict[int, DepthEntranceState],
    plane: Plane3D,
    plane_enter_direction: str,
    plane_hysteresis_mm: float,
    min_valid_pixels: int,
    roi_width_fraction: float,
    roi_height_fraction: float,
) -> DepthEntranceResult:
    entered_track_ids: list[int] = []
    depth_samples: Dict[int, DepthSample] = {}
    signed_distances_mm: Dict[int, float] = {}

    active_ids = {track.track_id for track in tracks if track.status != "REMOVED"}
    for track_id in list(states.keys()):
        if track_id not in active_ids:
            states.pop(track_id, None)

    for track in tracks:
        if track.status not in {"NEW", "TRACKED", "LOST"}:
            continue

        sample = sample_track_depth(
            depth_frame_mm,
            track,
            intrinsics=intrinsics,
            roi_width_fraction=roi_width_fraction,
            roi_height_fraction=roi_height_fraction,
            min_valid_pixels=min_valid_pixels,
        )
        if sample is None:
            continue

        depth_samples[track.track_id] = sample
        signed_distance_mm = signed_distance_to_plane_mm(sample.point_3d_mm, plane)
        signed_distances_mm[track.track_id] = signed_distance_mm

        state = states.setdefault(track.track_id, DepthEntranceState())
        state.recent_depths_mm.append(sample.depth_mm)
        state.recent_depths_mm = state.recent_depths_mm[-20:]

        if state.last_depth_mm is None:
            state.last_depth_mm = signed_distance_mm
            continue

        last_signed_distance_mm = state.last_depth_mm
        if plane_enter_direction == "positive_to_negative":
            crossed = last_signed_distance_mm > 0.0 and signed_distance_mm <= 0.0
            if not state.entered and crossed and track.status in {"TRACKED", "LOST"}:
                state.entered = True
                entered_track_ids.append(track.track_id)
            elif state.entered and signed_distance_mm >= plane_hysteresis_mm:
                state.entered = False
        else:
            crossed = last_signed_distance_mm < 0.0 and signed_distance_mm >= 0.0
            if not state.entered and crossed and track.status in {"TRACKED", "LOST"}:
                state.entered = True
                entered_track_ids.append(track.track_id)
            elif state.entered and signed_distance_mm <= -plane_hysteresis_mm:
                state.entered = False

        state.last_depth_mm = signed_distance_mm

    return DepthEntranceResult(
        entered_track_ids=entered_track_ids,
        depth_samples=depth_samples,
        signed_distances_mm=signed_distances_mm,
    )


def colorize_depth(depth_frame_mm: np.ndarray) -> np.ndarray:
    valid = depth_frame_mm[depth_frame_mm > 0]
    if valid.size == 0:
        return np.zeros((depth_frame_mm.shape[0], depth_frame_mm.shape[1], 3), dtype=np.uint8)

    near_mm = float(np.percentile(valid, 5))
    far_mm = float(np.percentile(valid, 95))
    if far_mm <= near_mm:
        far_mm = near_mm + 1.0

    clipped = np.clip(depth_frame_mm.astype(np.float32), near_mm, far_mm)
    normalized = ((clipped - near_mm) / (far_mm - near_mm) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[depth_frame_mm <= 0] = (0, 0, 0)
    return colored


def draw_depth_samples(
    frame: np.ndarray,
    *,
    tracks: Sequence[Track],
    depth_samples: Dict[int, DepthSample],
    depth_threshold_mm: float,
    signed_distances_mm: Dict[int, float] | None = None,
    plane_mode: bool = False,
) -> None:
    for track in tracks:
        sample = depth_samples.get(track.track_id)
        if sample is None:
            continue

        x1, y1, x2, y2 = sample.roi
        color = (0, 255, 0) if sample.depth_mm <= depth_threshold_mm else (0, 215, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if plane_mode and signed_distances_mm is not None and track.track_id in signed_distances_mm:
            signed_mm = signed_distances_mm[track.track_id]
            label = f"z={sample.depth_mm / 1000.0:.2f}m plane={signed_mm / 1000.0:+.2f}m"
        else:
            label = f"{sample.depth_mm / 1000.0:.2f}m"
        cv2.putText(
            frame,
            label,
            (track.x1, min(frame.shape[0] - 10, track.y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (track.x1, min(frame.shape[0] - 10, track.y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )

def draw_depth_event_banner(frame: np.ndarray, text: str) -> None:
    if not text:
        return

    height, width = frame.shape[:2]
    x1 = 20
    y1 = 50
    x2 = min(width - 20, width - 20)
    y2 = min(height - 20, 120)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 180), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        text,
        (x1 + 16, y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
