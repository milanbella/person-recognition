from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline import config as _pipeline_config  # noqa: F401
from pipeline.aruco_markers import (
    DEFAULT_ARUCO_DICTIONARY,
    DEFAULT_DOOR_MARKER_IDS,
    ArucoMarkerDetection,
    detect_aruco_markers,
    draw_aruco_detections,
    draw_rejected_aruco_candidates,
)
from pipeline.depth import (
    CameraIntrinsics,
    Plane3D,
    colorize_depth,
    fit_plane_from_points,
    orient_plane_normal_toward_positive_z,
    pixel_to_camera_point_mm,
    signed_distance_to_plane_mm,
)
from pipeline.rgbd_recording import (
    DEFAULT_PLANE_CALIBRATIONS_DIR,
    RGBDRecordingInfo,
    add_rgbd_recording_lookup_args,
    build_plane_calibration_path,
    load_depth_png,
    load_rgbd_recording,
    resolve_recording_dir,
)


MARKER_POSITION_BY_ID = {
    0: "upper_left",
    1: "upper_right",
    2: "lower_right",
    3: "lower_left",
}


@dataclass(frozen=True)
class ArucoPlanePoint:
    marker_id: int
    role: str
    center_px: tuple[float, float]
    sampled_pixel_xy: tuple[int, int]
    depth_mm: float
    valid_pixel_count: int
    point_3d_mm: tuple[float, float, float]


@dataclass(frozen=True)
class ArucoPlaneFit:
    plane: Plane3D
    rms_error_mm: float
    marker_points: list[ArucoPlanePoint]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a 3D door plane from detected ArUco markers in a recorded RGBD frame."
    )
    add_rgbd_recording_lookup_args(parser)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Initial frame index to show.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default=DEFAULT_ARUCO_DICTIONARY,
        help="OpenCV ArUco dictionary name.",
    )
    parser.add_argument(
        "--door-marker-id",
        type=int,
        nargs="*",
        default=list(DEFAULT_DOOR_MARKER_IDS),
        help="Door marker IDs to use for plane fitting.",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=8,
        help="Pixel radius around each marker center to search for valid depth.",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=5,
        help="Minimum valid depth pixels required around each marker center.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for fitted plane JSON. Defaults to plane_calibrations/plane_fit_<device-id>.json.",
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Draw rejected ArUco candidate quads in red.",
    )
    return parser


def build_output_json_path(
    *,
    device_id: str,
    explicit_path: Path | None,
    calibrations_root: Path = DEFAULT_PLANE_CALIBRATIONS_DIR,
) -> Path:
    if explicit_path is not None:
        explicit_path.parent.mkdir(parents=True, exist_ok=True)
        return explicit_path
    return build_plane_calibration_path(calibrations_root, device_id)


def load_rgb_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open RGB video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read RGB frame at index {frame_index}")
        return frame
    finally:
        capture.release()


def sample_depth_at_pixel(
    depth_frame_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    x: int,
    y: int,
    *,
    search_radius: int,
    min_valid_pixels: int,
) -> tuple[tuple[int, int], float, int, tuple[float, float, float]] | None:
    height, width = depth_frame_mm.shape[:2]
    x1 = max(0, x - search_radius)
    x2 = min(width, x + search_radius + 1)
    y1 = max(0, y - search_radius)
    y2 = min(height, y + search_radius + 1)
    roi = depth_frame_mm[y1:y2, x1:x2]

    valid_coords: list[tuple[int, int]] = []
    valid_depths: list[float] = []
    for local_y in range(roi.shape[0]):
        for local_x in range(roi.shape[1]):
            depth_value = float(roi[local_y, local_x])
            if depth_value > 0.0 and math.isfinite(depth_value):
                valid_coords.append((x1 + local_x, y1 + local_y))
                valid_depths.append(depth_value)

    if len(valid_depths) < min_valid_pixels:
        return None

    sampled_x, sampled_y = min(
        valid_coords,
        key=lambda pt: ((pt[0] - x) ** 2) + ((pt[1] - y) ** 2),
    )
    depth_mm = float(np.median(np.asarray(valid_depths, dtype=np.float32)))
    point_3d_mm = pixel_to_camera_point_mm(
        pixel_x=sampled_x,
        pixel_y=sampled_y,
        depth_mm=depth_mm,
        intrinsics=intrinsics,
    )
    return (sampled_x, sampled_y), depth_mm, len(valid_depths), point_3d_mm


def build_marker_points(
    *,
    detections: list[ArucoMarkerDetection],
    depth_frame_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    door_marker_ids: set[int],
    search_radius: int,
    min_valid_pixels: int,
) -> list[ArucoPlanePoint]:
    marker_points: list[ArucoPlanePoint] = []
    seen_marker_ids: set[int] = set()

    for detection in sorted(detections, key=lambda item: item.marker_id):
        if detection.marker_id not in door_marker_ids:
            continue
        if detection.marker_id in seen_marker_ids:
            continue

        center_x, center_y = detection.center_px
        sampled = sample_depth_at_pixel(
            depth_frame_mm,
            intrinsics,
            int(round(center_x)),
            int(round(center_y)),
            search_radius=search_radius,
            min_valid_pixels=min_valid_pixels,
        )
        if sampled is None:
            continue

        sampled_pixel_xy, depth_mm, valid_pixel_count, point_3d_mm = sampled
        marker_points.append(
            ArucoPlanePoint(
                marker_id=detection.marker_id,
                role=MARKER_POSITION_BY_ID.get(detection.marker_id, "door_marker"),
                center_px=detection.center_px,
                sampled_pixel_xy=sampled_pixel_xy,
                depth_mm=depth_mm,
                valid_pixel_count=valid_pixel_count,
                point_3d_mm=point_3d_mm,
            )
        )
        seen_marker_ids.add(detection.marker_id)

    return marker_points


def fit_plane(marker_points: list[ArucoPlanePoint]) -> ArucoPlaneFit | None:
    if len(marker_points) < 3:
        return None

    raw_plane = fit_plane_from_points([point.point_3d_mm for point in marker_points])
    plane = orient_plane_normal_toward_positive_z(raw_plane)
    distances = [
        signed_distance_to_plane_mm(point.point_3d_mm, plane)
        for point in marker_points
    ]
    rms_error_mm = math.sqrt(sum(value * value for value in distances) / len(distances))
    return ArucoPlaneFit(
        plane=plane,
        rms_error_mm=rms_error_mm,
        marker_points=marker_points,
    )


def build_output_payload(
    *,
    recording: RGBDRecordingInfo,
    frame_index: int,
    dictionary_name: str,
    fit: ArucoPlaneFit,
) -> dict:
    plane = fit.plane
    return {
        "schema_version": "2026-06-14",
        "calibration_method": "aruco_markers",
        "recording_dir": str(recording.recording_dir.resolve()),
        "device_id": recording.device_id,
        "frame_index": frame_index,
        "rgb_host_synced_seconds": recording.frames[frame_index].rgb_host_synced_seconds,
        "aruco_dictionary": dictionary_name,
        "marker_positions": {str(key): value for key, value in MARKER_POSITION_BY_ID.items()},
        "marker_points": [
            {
                "marker_id": point.marker_id,
                "role": point.role,
                "center_px": [point.center_px[0], point.center_px[1]],
                "sampled_pixel_xy": list(point.sampled_pixel_xy),
                "depth_mm": point.depth_mm,
                "valid_pixel_count": point.valid_pixel_count,
                "point_3d_mm": list(point.point_3d_mm),
            }
            for point in fit.marker_points
        ],
        "plane_point_mm": list(plane.point_mm),
        "plane_normal": list(plane.normal),
        "rms_fit_error_mm": fit.rms_error_mm,
        "recommended_enter_direction_if_person_moves_toward_camera": "positive_to_negative",
        "recommended_replay_cli": (
            f"python .\\replay_depth_tuner.py "
            f"--device-id {recording.device_id} "
            f"--depth-trigger-mode plane"
        ),
    }


def draw_marker_points(frame: np.ndarray, marker_points: list[ArucoPlanePoint]) -> None:
    for point in marker_points:
        x, y = point.sampled_pixel_xy
        cv2.circle(frame, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        label = f"{point.marker_id} {point.role} z={point.depth_mm / 1000.0:.2f}m"
        cv2.putText(
            frame,
            label,
            (x + 12, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (x + 12, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_status_overlay(
    frame: np.ndarray,
    *,
    device_id: str,
    frame_index: int,
    detected_ids: list[int],
    marker_points: list[ArucoPlanePoint],
    fit: ArucoPlaneFit | None,
    output_json_path: Path,
) -> None:
    valid_ids = [point.marker_id for point in marker_points]
    if fit is None:
        fit_text = f"fit=not ready valid_points={len(marker_points)}/3"
    else:
        fit_text = f"fit=ready rms={fit.rms_error_mm:.2f}mm valid_points={len(marker_points)}"

    lines = [
        f"device={device_id} frame={frame_index}",
        f"detected={detected_ids} valid_depth_ids={valid_ids}",
        fit_text,
        "f/enter=save, a/d=prev/next, j/l=-10/+10, q=quit",
        f"output={output_json_path}",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28


def render_depth_frame(
    depth_frame_mm: np.ndarray,
    marker_points: list[ArucoPlanePoint],
) -> np.ndarray:
    colored = colorize_depth(depth_frame_mm)
    for point in marker_points:
        x, y = point.sampled_pixel_xy
        cv2.circle(colored, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(colored, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            colored,
            str(point.marker_id),
            (x + 10, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return colored


def main() -> None:
    args = build_argparser().parse_args()
    recording_dir = resolve_recording_dir(
        recording_dir=args.recording_dir,
        device_id=args.device_id,
        recordings_root=args.recordings_root,
    )
    recording = load_rgbd_recording(recording_dir)
    if recording.rgb_intrinsics is None:
        raise RuntimeError(
            "This RGBD recording does not contain RGB intrinsics. Re-record with the current record_rgbd_stream.py."
        )

    intrinsics = CameraIntrinsics(
        fx=float(recording.rgb_intrinsics["fx"]),
        fy=float(recording.rgb_intrinsics["fy"]),
        cx=float(recording.rgb_intrinsics["cx"]),
        cy=float(recording.rgb_intrinsics["cy"]),
    )
    door_marker_ids = set(args.door_marker_id)
    output_json_path = build_output_json_path(
        device_id=recording.device_id,
        explicit_path=args.output_json,
    )

    frame_index = max(0, min(args.frame_index, len(recording.frames) - 1))
    current_rgb = load_rgb_frame(recording.rgb_video_path, frame_index)
    current_depth = load_depth_png(recording, recording.frames[frame_index])

    cv2.namedWindow("ArUco Plane Fit RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ArUco Plane Fit Depth", cv2.WINDOW_NORMAL)

    print(f"Loaded recording: {recording.recording_dir}")
    print(f"Dictionary: {args.dictionary}")
    print(f"Door marker IDs: {sorted(door_marker_ids)}")
    print("Controls: a/d previous/next frame, j/l -10/+10 frames, f/enter fit+save, q quit.")

    def reload_frame(new_index: int) -> None:
        nonlocal frame_index, current_rgb, current_depth
        frame_index = max(0, min(new_index, len(recording.frames) - 1))
        current_rgb = load_rgb_frame(recording.rgb_video_path, frame_index)
        current_depth = load_depth_png(recording, recording.frames[frame_index])

    try:
        while True:
            result = detect_aruco_markers(current_rgb, dictionary_name=args.dictionary)
            marker_points = build_marker_points(
                detections=result.detections,
                depth_frame_mm=current_depth,
                intrinsics=intrinsics,
                door_marker_ids=door_marker_ids,
                search_radius=args.search_radius,
                min_valid_pixels=args.min_valid_pixels,
            )
            fit = fit_plane(marker_points)

            rgb_view = current_rgb.copy()
            draw_aruco_detections(
                rgb_view,
                result.detections,
                door_marker_ids=door_marker_ids,
            )
            if args.show_rejected:
                draw_rejected_aruco_candidates(rgb_view, result.rejected_candidates)
            draw_marker_points(rgb_view, marker_points)
            draw_status_overlay(
                rgb_view,
                device_id=recording.device_id,
                frame_index=frame_index,
                detected_ids=sorted(detection.marker_id for detection in result.detections),
                marker_points=marker_points,
                fit=fit,
                output_json_path=output_json_path,
            )

            cv2.imshow("ArUco Plane Fit RGB", rgb_view)
            cv2.imshow("ArUco Plane Fit Depth", render_depth_frame(current_depth, marker_points))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                reload_frame(frame_index - 1)
                continue
            if key == ord("d"):
                reload_frame(frame_index + 1)
                continue
            if key == ord("j"):
                reload_frame(frame_index - 10)
                continue
            if key == ord("l"):
                reload_frame(frame_index + 10)
                continue
            if key == ord("f") or key == 13:
                if fit is None:
                    print(
                        "Need at least 3 detected door markers with valid depth before saving. "
                        f"Current valid count: {len(marker_points)}"
                    )
                    continue

                payload = build_output_payload(
                    recording=recording,
                    frame_index=frame_index,
                    dictionary_name=args.dictionary,
                    fit=fit,
                )
                output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                plane = fit.plane
                print("\nFitted ArUco plane:")
                print(
                    f"  point_mm = ({plane.point_mm[0]:.1f}, {plane.point_mm[1]:.1f}, {plane.point_mm[2]:.1f})"
                )
                print(
                    f"  normal   = ({plane.normal[0]:.5f}, {plane.normal[1]:.5f}, {plane.normal[2]:.5f})"
                )
                print(f"  rms_fit_error_mm = {fit.rms_error_mm:.2f}")
                print(f"  marker_ids = {[point.marker_id for point in fit.marker_points]}")
                print("\nRecommended replay command:")
                print(payload["recommended_replay_cli"])
                print(f"\nSaved plane fit to {output_json_path}\n")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
