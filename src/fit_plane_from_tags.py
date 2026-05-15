import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.depth import (
    CameraIntrinsics,
    colorize_depth,
    fit_plane_from_points,
    orient_plane_normal_toward_positive_z,
    pixel_to_camera_point_mm,
    signed_distance_to_plane_mm,
)
from pipeline.rgbd_recording import (
    DEFAULT_PLANE_CALIBRATIONS_DIR,
    add_rgbd_recording_lookup_args,
    build_plane_calibration_path,
    load_depth_png,
    load_rgbd_recording,
    resolve_recording_dir,
)


@dataclass
class ClickedPoint:
    pixel_xy: tuple[int, int]
    sampled_pixel_xy: tuple[int, int]
    depth_mm: float
    point_3d_mm: tuple[float, float, float]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a 3D door plane from 3 manually clicked points on a recorded RGBD frame."
    )
    add_rgbd_recording_lookup_args(parser)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Initial frame index to show.",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=6,
        help="Pixel radius around each click to search for valid depth.",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=5,
        help="Minimum valid depth pixels required in the search neighborhood.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for fitted plane JSON. Defaults inside the recording dir.",
    )
    return parser


def build_output_json_path(
    *,
    device_id: str,
    explicit_path: Path | None,
    calibrations_root: Path = DEFAULT_PLANE_CALIBRATIONS_DIR,
) -> Path:
    if explicit_path is not None:
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


def sample_clicked_point(
    depth_frame_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    x: int,
    y: int,
    *,
    search_radius: int,
    min_valid_pixels: int,
) -> ClickedPoint | None:
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

    return ClickedPoint(
        pixel_xy=(x, y),
        sampled_pixel_xy=(sampled_x, sampled_y),
        depth_mm=depth_mm,
        point_3d_mm=point_3d_mm,
    )


def render_rgb_frame(
    frame: np.ndarray,
    *,
    frame_index: int,
    device_id: str,
    clicked_points: list[ClickedPoint],
) -> np.ndarray:
    overlay = frame.copy()
    for idx, point in enumerate(clicked_points, start=1):
        x, y = point.pixel_xy
        sx, sy = point.sampled_pixel_xy
        cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (sx, sy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        text = f"{idx}: z={point.depth_mm/1000.0:.2f}m"
        cv2.putText(
            overlay,
            text,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    header_lines = [
        f"device={device_id} frame={frame_index}",
        "click 3 tagged door points, c=clear, a/d=prev/next, j/l=-10/+10, f=fit, q=quit",
    ]
    y = 28
    for line in header_lines:
        cv2.putText(
            overlay,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28

    return overlay


def render_depth_frame(
    depth_frame_mm: np.ndarray,
    clicked_points: list[ClickedPoint],
) -> np.ndarray:
    colored = colorize_depth(depth_frame_mm)
    for idx, point in enumerate(clicked_points, start=1):
        x, y = point.pixel_xy
        sx, sy = point.sampled_pixel_xy
        cv2.circle(colored, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(colored, (sx, sy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            colored,
            str(idx),
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

    frame_index = max(0, min(args.frame_index, len(recording.frames) - 1))
    clicked_points: list[ClickedPoint] = []
    output_json_path = build_output_json_path(
        device_id=recording.device_id,
        explicit_path=args.output_json,
    )

    current_rgb = load_rgb_frame(recording.rgb_video_path, frame_index)
    current_depth = load_depth_png(recording, recording.frames[frame_index])

    def reload_frame(new_index: int) -> None:
        nonlocal frame_index, current_rgb, current_depth, clicked_points
        frame_index = max(0, min(new_index, len(recording.frames) - 1))
        current_rgb = load_rgb_frame(recording.rgb_video_path, frame_index)
        current_depth = load_depth_png(recording, recording.frames[frame_index])
        clicked_points = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal clicked_points
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(clicked_points) >= 3:
            return
        sampled = sample_clicked_point(
            current_depth,
            intrinsics,
            x,
            y,
            search_radius=args.search_radius,
            min_valid_pixels=args.min_valid_pixels,
        )
        if sampled is None:
            print(
                f"No valid depth near clicked point ({x}, {y}). Try clicking closer to the tag edge or increase --search-radius."
            )
            return
        clicked_points.append(sampled)
        print(
            f"Point {len(clicked_points)}: px={sampled.pixel_xy} sampled={sampled.sampled_pixel_xy} "
            f"xyz_mm=({sampled.point_3d_mm[0]:.1f}, {sampled.point_3d_mm[1]:.1f}, {sampled.point_3d_mm[2]:.1f})"
        )

    cv2.namedWindow("Plane Fit RGB", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Plane Fit RGB", on_mouse)

    print(f"Loaded recording: {recording.recording_dir}")
    print("Click 3 door-corner tags in the RGB window.")
    print("Controls: a/d previous/next frame, j/l -10/+10 frames, c clear, f fit, q quit.")

    try:
        while True:
            rgb_view = render_rgb_frame(
                current_rgb,
                frame_index=frame_index,
                device_id=recording.device_id,
                clicked_points=clicked_points,
            )
            depth_view = render_depth_frame(current_depth, clicked_points)

            cv2.imshow("Plane Fit RGB", rgb_view)
            cv2.imshow("Plane Fit Depth", depth_view)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                clicked_points = []
                continue
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
                if len(clicked_points) < 3:
                    print("Need exactly 3 clicked points to fit a plane.")
                    continue

                raw_plane = fit_plane_from_points([point.point_3d_mm for point in clicked_points])
                plane = orient_plane_normal_toward_positive_z(raw_plane)
                distances = [
                    signed_distance_to_plane_mm(point.point_3d_mm, plane)
                    for point in clicked_points
                ]
                rms_error_mm = math.sqrt(sum(value * value for value in distances) / len(distances))

                output_payload = {
                    "schema_version": "2026-05-11",
                    "recording_dir": str(recording.recording_dir.resolve()),
                    "device_id": recording.device_id,
                    "frame_index": frame_index,
                    "rgb_host_synced_seconds": recording.frames[frame_index].rgb_host_synced_seconds,
                    "clicked_points": [
                        {
                            "pixel_xy": list(point.pixel_xy),
                            "sampled_pixel_xy": list(point.sampled_pixel_xy),
                            "depth_mm": point.depth_mm,
                            "point_3d_mm": list(point.point_3d_mm),
                        }
                        for point in clicked_points
                    ],
                    "plane_point_mm": list(plane.point_mm),
                    "plane_normal": list(plane.normal),
                    "rms_fit_error_mm": rms_error_mm,
                    "recommended_enter_direction_if_person_moves_toward_camera": "positive_to_negative",
                    "recommended_replay_cli": (
                        f"python .\\replay_depth_tuner.py "
                        f"--device-id {recording.device_id} "
                        f"--depth-trigger-mode plane"
                    ),
                    "cli_args": (
                        f"--depth-trigger-mode plane "
                        f"--plane-point-x-mm {plane.point_mm[0]:.1f} "
                        f"--plane-point-y-mm {plane.point_mm[1]:.1f} "
                        f"--plane-point-z-mm {plane.point_mm[2]:.1f} "
                        f"--plane-normal-x {plane.normal[0]:.5f} "
                        f"--plane-normal-y {plane.normal[1]:.5f} "
                        f"--plane-normal-z {plane.normal[2]:.5f} "
                        f"--plane-enter-direction positive_to_negative"
                    ),
                }
                output_json_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

                print("\nFitted plane:")
                print(
                    f"  point_mm = ({plane.point_mm[0]:.1f}, {plane.point_mm[1]:.1f}, {plane.point_mm[2]:.1f})"
                )
                print(
                    f"  normal   = ({plane.normal[0]:.5f}, {plane.normal[1]:.5f}, {plane.normal[2]:.5f})"
                )
                print(f"  rms_fit_error_mm = {rms_error_mm:.2f}")
                print("\nRecommended replay command:")
                print(output_payload["recommended_replay_cli"])
                print("\nRecommended CLI fragment:")
                print(output_payload["cli_args"])
                print(f"\nSaved plane fit to {output_json_path}\n")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
