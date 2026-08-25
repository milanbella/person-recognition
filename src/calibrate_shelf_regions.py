from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import cv2
import depthai as dai
import numpy as np

from pipeline.camera import configure_live_device, print_connected_device
from pipeline.depth import CameraIntrinsics, intrinsics_from_matrix
from pipeline.shelf_config import (
    DEFAULT_SHELF_CALIBRATIONS_DIR,
    DEFAULT_SHELF_CONFIG_PATH,
    load_shelf_config,
)
from pipeline.shelf_regions import (
    CameraShelfRegions,
    NormalizedPoint,
    ShelfRegion,
    SHELF_REGION_SCHEMA_VERSION,
    build_shelf_depth_model,
    depth_calibration_identity,
    load_camera_shelf_regions,
    normalized_point_in_polygon,
    save_camera_shelf_regions,
    shelf_regions_path,
)


WINDOW_NAME = "Shelf Region Calibration"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw shelf polygons and calibrate a multi-frame RGB-aligned 3D "
            "depth model for each shelf."
        )
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--calibration-frames", type=int, default=60)
    parser.add_argument("--grid-columns", type=int, default=24)
    parser.add_argument("--grid-rows", type=int, default=16)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.35)
    parser.add_argument("--minimum-valid-samples", type=int, default=100)
    parser.add_argument("--minimum-depth-mm", type=int, default=200)
    parser.add_argument("--maximum-depth-mm", type=int, default=12000)
    parser.add_argument("--off-shelf-hysteresis-mm", type=float, default=150.0)
    parser.add_argument(
        "--inspect-saved",
        action="store_true",
        help=(
            "Capture fresh RGB/depth frames, compare them with the saved 3D "
            "calibration, write diagnostic PNG files, and exit."
        ),
    )
    parser.add_argument("--inspection-frames", type=int, default=15)
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="Diagnostic output directory. Default: <calibration-root>/region_previews.",
    )
    parser.add_argument("--shelf-config", type=Path, default=DEFAULT_SHELF_CONFIG_PATH)
    parser.add_argument("--shelf-id", type=int, nargs="+", default=None)
    parser.add_argument(
        "--shelf-calibrations-root",
        type=Path,
        default=DEFAULT_SHELF_CALIBRATIONS_DIR,
    )
    parser.add_argument("--depth-tolerance-mm", type=float, default=450.0)
    return parser


def _open_live_frame(
    stack: ExitStack, args: argparse.Namespace
) -> tuple[
    dai.Device,
    dai.Pipeline,
    dai.MessageQueue,
    dai.MessageQueue,
    CameraIntrinsics,
    str,
]:
    device = dai.Device(args.device_id)
    configure_live_device(device)
    print_connected_device(device)
    calibration = device.readCalibration()
    intrinsics = intrinsics_from_matrix(
        calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            (args.depth_width, args.depth_height),
        )
    )
    intrinsics_tuple = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    calibration_id = depth_calibration_identity(
        args.device_id,
        width=args.depth_width,
        height=args.depth_height,
        intrinsics=intrinsics_tuple,
    )
    pipeline = stack.enter_context(dai.Pipeline(device))
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = camera.requestOutput(
        size=(args.width, args.height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=args.fps,
    )
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(args.depth_width, args.depth_height)
    mono_left.requestFullResolutionOutput(fps=args.fps).link(stereo.left)
    mono_right.requestFullResolutionOutput(fps=args.fps).link(stereo.right)
    queue = output.createOutputQueue(maxSize=8, blocking=False)
    depth_queue = stereo.depth.createOutputQueue(maxSize=8, blocking=False)
    pipeline.start()
    return device, pipeline, queue, depth_queue, intrinsics, calibration_id


def _draw(
    frame: np.ndarray,
    *,
    shelf_ids: list[int],
    shelf_index: int,
    current_points: list[tuple[int, int]],
    saved_polygons: dict[int, list[list[tuple[int, int]]]],
    frozen: bool,
) -> np.ndarray:
    panel_width = 620
    frame_height, frame_width = frame.shape[:2]
    preview = np.full(
        (frame_height, frame_width + panel_width, 3),
        (20, 20, 20),
        dtype=frame.dtype,
    )
    preview[:, :frame_width] = frame
    colors = [(39, 220, 255), (80, 220, 120), (255, 120, 80), (220, 80, 220)]
    for index, shelf_id in enumerate(shelf_ids):
        color = colors[index % len(colors)]
        polygons = list(saved_polygons.get(shelf_id, []))
        if index == shelf_index and current_points:
            polygons.append(current_points)
        for polygon_index, points in enumerate(polygons):
            for point in points:
                cv2.circle(preview, point, 6, color, -1, cv2.LINE_AA)
            if len(points) >= 2:
                cv2.polylines(
                    preview,
                    [np.asarray(points, dtype=np.int32)],
                    len(points) >= 3,
                    color,
                    3,
                    cv2.LINE_AA,
                )
            cv2.putText(
                preview,
                f"Shelf {shelf_id}",
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"face {polygon_index + 1}",
                (points[0][0], points[0][1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
    completed_surfaces = len(saved_polygons.get(shelf_ids[shelf_index], []))
    status = (
        f"Shelf {shelf_ids[shelf_index]} ({shelf_index + 1}/{len(shelf_ids)}) "
        f"surfaces={completed_surfaces} current_points={len(current_points)} "
        f"{'FROZEN' if frozen else 'LIVE'}"
    )
    panel_x = frame_width + 20
    cv2.putText(
        preview,
        status,
        (panel_x, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    controls = (
        "Space  freeze / resume",
        "Click  add polygon point",
        "p      complete this surface",
        "x      remove last surface",
        "r      reset current points",
        "n      next shelf",
        "s      capture depth and save",
        "q      quit without saving",
    )
    for index, control in enumerate(controls):
        cv2.putText(
            preview,
            control,
            (panel_x, 90 + index * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return preview


def _collect_synchronized_depth_frames(
    *,
    rgb_queue: dai.MessageQueue,
    depth_queue: dai.MessageQueue,
    frame_count: int,
    fallback_frame: np.ndarray,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    rgb_by_sequence: dict[int, object] = {}
    depth_by_sequence: dict[int, object] = {}
    depth_frames: list[np.ndarray] = []
    latest_frame = fallback_frame
    last_sequence = -1
    print(
        f"Capturing {frame_count} synchronized shelf depth frames; keep shelves "
        "unoccluded. Press q to cancel."
    )
    while len(depth_frames) < frame_count:
        rgb_message = rgb_queue.tryGet()
        while rgb_message is not None:
            rgb_by_sequence[int(rgb_message.getSequenceNum())] = rgb_message
            rgb_message = rgb_queue.tryGet()
        depth_message = depth_queue.tryGet()
        while depth_message is not None:
            depth_by_sequence[int(depth_message.getSequenceNum())] = depth_message
            depth_message = depth_queue.tryGet()
        common = sorted(
            sequence
            for sequence in set(rgb_by_sequence) & set(depth_by_sequence)
            if sequence > last_sequence
        )
        for sequence in common:
            rgb_message = rgb_by_sequence.pop(sequence)
            depth_message = depth_by_sequence.pop(sequence)
            latest_frame = rgb_message.getCvFrame()
            depth_frames.append(depth_message.getFrame().copy())
            last_sequence = sequence
            if len(depth_frames) >= frame_count:
                break
        rgb_by_sequence = {
            sequence: message
            for sequence, message in rgb_by_sequence.items()
            if sequence > last_sequence
        }
        depth_by_sequence = {
            sequence: message
            for sequence, message in depth_by_sequence.items()
            if sequence > last_sequence
        }
        if show_progress:
            progress = latest_frame.copy()
            cv2.rectangle(progress, (0, 0), (progress.shape[1], 46), (20, 20, 20), -1)
            cv2.putText(
                progress,
                f"3D depth capture {len(depth_frames)}/{frame_count}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, progress)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                raise KeyboardInterrupt("Shelf depth capture cancelled.")
        if not common:
            time.sleep(0.005)
    return depth_frames, latest_frame


def _preview_depth_models(
    frame: np.ndarray,
    *,
    regions: tuple[ShelfRegion, ...],
    screenshot_path: Path | None = None,
) -> bool:
    preview = frame.copy()
    height, width = preview.shape[:2]
    all_depths = [
        cell.median_depth_mm
        for region in regions
        if region.depth_model is not None
        for cell in region.depth_model.cells
    ]
    minimum = min(all_depths)
    maximum = max(all_depths)
    span = max(1.0, maximum - minimum)
    for region in regions:
        model = region.depth_model
        if model is None:
            continue
        valid_cells = {(cell.column, cell.row): cell for cell in model.cells}
        eligible_cells = 0
        for row in range(model.rows):
            for column in range(model.columns):
                if not any(
                    normalized_point_in_polygon(
                        (column + 0.5) / model.columns,
                        (row + 0.5) / model.rows,
                        polygon,
                    )
                    for polygon in region.polygons
                ):
                    continue
                eligible_cells += 1
                cell = valid_cells.get((column, row))
                x1 = int(round(column * width / model.columns))
                x2 = int(round((column + 1) * width / model.columns))
                y1 = int(round(row * height / model.rows))
                y2 = int(round((row + 1) * height / model.rows))
                if cell is None:
                    color = (40, 40, 180)
                else:
                    ratio = (cell.median_depth_mm - minimum) / span
                    color = (int(255 * ratio), int(255 * (1.0 - ratio)), 40)
                overlay = preview.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0.0)
        polygon_points = [
            np.asarray(
                [(int(point.x * width), int(point.y * height)) for point in polygon],
                dtype=np.int32,
            )
            for polygon in region.polygons
        ]
        cv2.polylines(preview, polygon_points, True, (39, 220, 255), 3, cv2.LINE_AA)
        eligible_cells = max(eligible_cells, len(model.cells))
        coverage = len(model.cells) / max(1, eligible_cells)
        cv2.putText(
            preview,
            f"Shelf {region.shelf_id} depth cells {coverage:.0%}",
            tuple(polygon_points[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(
            f"SHELF_3D_CALIBRATION shelf_id={region.shelf_id} "
            f"valid_cells={len(model.cells)}/{eligible_cells} "
            f"coverage={coverage:.3f} frames={model.calibration_frame_count}"
        )
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 48), (20, 20, 20), -1)
    cv2.putText(
        preview,
        "3D depth preview: s confirm/save | q cancel",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if screenshot_path is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(screenshot_path), preview):
            raise RuntimeError(f"Could not write calibration preview: {screenshot_path}")
        print(f"Saved calibration preview screenshot to {screenshot_path}")
    while True:
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            return True
        if key == ord("q"):
            return False


def _write_saved_calibration_inspection(
    *,
    args: argparse.Namespace,
    config: CameraShelfRegions,
    rgb_frame: np.ndarray,
    depth_frames: list[np.ndarray],
    calibration_id: str,
) -> tuple[Path, Path, Path]:
    screenshot_dir = args.screenshot_dir or (
        args.shelf_calibrations_root / "region_previews"
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = screenshot_dir / f"shelf_regions_{args.device_id}_{timestamp}"
    raw_path = prefix.with_name(f"{prefix.name}_rgb.png")
    residual_path = prefix.with_name(f"{prefix.name}_residual.png")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    if not cv2.imwrite(str(raw_path), rgb_frame):
        raise RuntimeError(f"Could not write inspection RGB screenshot: {raw_path}")

    overlay = rgb_frame.copy()
    raw_height, raw_width = rgb_frame.shape[:2]
    shelf_summaries: list[dict[str, object]] = []
    for region in config.regions:
        model = region.depth_model
        if model is None:
            continue
        if model.camera_calibration_id != calibration_id:
            raise ValueError(
                f"Shelf {region.shelf_id} calibration identity does not match "
                "the current camera configuration."
            )
        if any(
            frame.shape != (model.aligned_depth_height, model.aligned_depth_width)
            for frame in depth_frames
        ):
            raise ValueError(
                f"Shelf {region.shelf_id} expects depth "
                f"{model.aligned_depth_width}x{model.aligned_depth_height}."
            )
        matched = 0
        transition = 0
        mismatched = 0
        invalid = 0
        residuals: list[float] = []
        depth_mask = np.zeros(
            (model.aligned_depth_height, model.aligned_depth_width),
            dtype=np.uint8,
        )
        depth_polygon_points = [
            np.asarray(
                [
                    (
                        int(round(point.x * model.aligned_depth_width)),
                        int(round(point.y * model.aligned_depth_height)),
                    )
                    for point in polygon
                ],
                dtype=np.int32,
            )
            for polygon in region.polygons
        ]
        cv2.fillPoly(depth_mask, depth_polygon_points, 1)
        for cell in model.cells:
            depth_x1 = int(round(cell.column * model.aligned_depth_width / model.columns))
            depth_x2 = int(round((cell.column + 1) * model.aligned_depth_width / model.columns))
            depth_y1 = int(round(cell.row * model.aligned_depth_height / model.rows))
            depth_y2 = int(round((cell.row + 1) * model.aligned_depth_height / model.rows))
            samples = []
            cell_mask = depth_mask[depth_y1:depth_y2, depth_x1:depth_x2].astype(
                bool,
                copy=False,
            )
            for frame in depth_frames:
                values = frame[depth_y1:depth_y2, depth_x1:depth_x2][cell_mask]
                valid = values[(values > 0) & np.isfinite(values)]
                if valid.size:
                    samples.append(valid.astype(np.float32, copy=False))
            if samples:
                current_depth = float(np.median(np.concatenate(samples)))
                residual = current_depth - cell.median_depth_mm
                residuals.append(residual)
                tolerance = max(
                    region.depth_tolerance_mm or 250.0,
                    3.0 * cell.mad_depth_mm,
                )
                if abs(residual) <= tolerance:
                    color = (60, 190, 70)
                    matched += 1
                elif abs(residual) <= tolerance + region.off_shelf_hysteresis_mm:
                    color = (30, 170, 240)
                    transition += 1
                else:
                    color = (40, 40, 220)
                    mismatched += 1
            else:
                color = (100, 100, 100)
                invalid += 1
            x1 = int(round(cell.column * raw_width / model.columns))
            x2 = int(round((cell.column + 1) * raw_width / model.columns))
            y1 = int(round(cell.row * raw_height / model.rows))
            y2 = int(round((cell.row + 1) * raw_height / model.rows))
            cell_overlay = overlay.copy()
            cv2.rectangle(cell_overlay, (x1, y1), (x2, y2), color, -1)
            overlay = cv2.addWeighted(cell_overlay, 0.28, overlay, 0.72, 0.0)
        polygon_points = [
            np.asarray(
                [(int(point.x * raw_width), int(point.y * raw_height)) for point in polygon],
                dtype=np.int32,
            )
            for polygon in region.polygons
        ]
        cv2.polylines(overlay, polygon_points, True, (39, 220, 255), 3, cv2.LINE_AA)
        label_at = tuple(polygon_points[0][0])
        cv2.putText(
            overlay,
            f"Shelf {region.shelf_id}: green={matched} orange={transition} red={mismatched} gray={invalid}",
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        absolute_residuals = sorted(abs(value) for value in residuals)
        shelf_summaries.append(
            {
                "shelfId": region.shelf_id,
                "surfaceCount": len(region.polygons),
                "calibratedCellCount": len(model.cells),
                "matchedCellCount": matched,
                "transitionCellCount": transition,
                "mismatchedCellCount": mismatched,
                "invalidCellCount": invalid,
                "medianAbsoluteResidualMm": (
                    None
                    if not absolute_residuals
                    else absolute_residuals[len(absolute_residuals) // 2]
                ),
                "maximumAbsoluteResidualMm": (
                    None if not absolute_residuals else absolute_residuals[-1]
                ),
            }
        )
    if not cv2.imwrite(str(residual_path), overlay):
        raise RuntimeError(f"Could not write inspection residual screenshot: {residual_path}")
    summary_path.write_text(
        json.dumps(
            {
                "schemaVersion": config.schema_version,
                "deviceId": config.device_id,
                "inspectionFrameCount": len(depth_frames),
                "shelves": shelf_summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return raw_path, residual_path, summary_path


def _inspect_saved_calibration(args: argparse.Namespace) -> None:
    path = shelf_regions_path(args.shelf_calibrations_root, args.device_id)
    config = load_camera_shelf_regions(path)
    if not config.has_3d_calibration:
        raise ValueError(f"Saved calibration is not schema-version-2 3D: {path}")
    with ExitStack() as stack:
        (
            _device,
            _pipeline,
            rgb_queue,
            depth_queue,
            _intrinsics,
            calibration_id,
        ) = _open_live_frame(stack, args)
        empty = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        depth_frames, rgb_frame = _collect_synchronized_depth_frames(
            rgb_queue=rgb_queue,
            depth_queue=depth_queue,
            frame_count=args.inspection_frames,
            fallback_frame=empty,
            show_progress=False,
        )
    raw_path, residual_path, summary_path = _write_saved_calibration_inspection(
        args=args,
        config=config,
        rgb_frame=rgb_frame,
        depth_frames=depth_frames,
        calibration_id=calibration_id,
    )
    print(f"Saved inspection RGB screenshot to {raw_path}")
    print(f"Saved inspection residual screenshot to {residual_path}")
    print(f"Saved inspection summary to {summary_path}")


def main() -> None:
    args = build_argparser().parse_args()
    if min(args.width, args.height, args.depth_width, args.depth_height, args.fps) <= 0:
        raise ValueError("Frame dimensions and FPS must be positive.")
    if args.width * args.depth_height != args.height * args.depth_width:
        raise ValueError("RGB and aligned depth must use the same aspect ratio.")
    if args.depth_tolerance_mm <= 0.0:
        raise ValueError("--depth-tolerance-mm must be positive.")
    if args.off_shelf_hysteresis_mm <= 0.0:
        raise ValueError("--off-shelf-hysteresis-mm must be positive.")
    if min(args.calibration_frames, args.grid_columns, args.grid_rows) <= 0:
        raise ValueError("Calibration frame and grid counts must be positive.")
    if args.inspection_frames <= 0:
        raise ValueError("--inspection-frames must be positive.")
    if not 0.0 < args.minimum_valid_fraction <= 1.0:
        raise ValueError("--minimum-valid-fraction must be between zero and one.")
    if args.minimum_depth_mm <= 0 or args.maximum_depth_mm <= args.minimum_depth_mm:
        raise ValueError("Invalid calibration depth range.")
    if args.inspect_saved:
        if args.image is not None:
            raise ValueError("--inspect-saved cannot be combined with --image.")
        _inspect_saved_calibration(args)
        return
    config = load_shelf_config(args.shelf_config)
    available_ids = [shelf.shelf_id for shelf in config.shelves]
    shelf_ids = available_ids if args.shelf_id is None else args.shelf_id
    unknown = sorted(set(shelf_ids) - set(available_ids))
    if unknown:
        raise ValueError(f"Unknown shelf IDs: {unknown}")
    if len(set(shelf_ids)) != len(shelf_ids) or not shelf_ids:
        raise ValueError("Shelf IDs must be non-empty and unique.")

    with ExitStack() as stack:
        queue = None
        depth_queue = None
        intrinsics = None
        calibration_id = None
        if args.image is not None:
            base_frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if base_frame is None:
                raise ValueError(f"Could not read calibration image: {args.image}")
            frozen = True
        else:
            (
                _device,
                _pipeline,
                queue,
                depth_queue,
                intrinsics,
                calibration_id,
            ) = _open_live_frame(stack, args)
            base_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            frozen = False

        shelf_index = 0
        current_points: list[tuple[int, int]] = []
        saved_polygons: dict[int, list[list[tuple[int, int]]]] = {}

        def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
            frame_height, frame_width = base_frame.shape[:2]
            if (
                event == cv2.EVENT_LBUTTONDOWN
                and 0 <= x < frame_width
                and 0 <= y < frame_height
            ):
                current_points.append((x, y))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        print("Shelf region calibration started. q exits without saving.")
        while True:
            if queue is not None and not frozen:
                message = queue.tryGet()
                newer = queue.tryGet()
                while newer is not None:
                    message = newer
                    newer = queue.tryGet()
                if message is not None:
                    base_frame = message.getCvFrame()
            cv2.imshow(
                WINDOW_NAME,
                _draw(
                    base_frame,
                    shelf_ids=shelf_ids,
                    shelf_index=shelf_index,
                    current_points=current_points,
                    saved_polygons=saved_polygons,
                    frozen=frozen,
                ),
            )
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                print("Shelf region calibration cancelled; nothing saved.")
                break
            if key == ord(" ") and queue is not None:
                frozen = not frozen
            elif key == ord("r"):
                current_points.clear()
            elif key == ord("p"):
                if len(current_points) < 3:
                    print("A shelf surface requires at least three points.")
                    continue
                shelf_id = shelf_ids[shelf_index]
                saved_polygons.setdefault(shelf_id, []).append(list(current_points))
                current_points.clear()
                print(
                    f"Added surface {len(saved_polygons[shelf_id])} "
                    f"for shelf {shelf_id}."
                )
            elif key == ord("x"):
                shelf_id = shelf_ids[shelf_index]
                polygons = saved_polygons.get(shelf_id, [])
                if polygons:
                    polygons.pop()
                    print(f"Removed last completed surface for shelf {shelf_id}.")
            elif key == ord("n"):
                shelf_id = shelf_ids[shelf_index]
                if current_points and len(current_points) < 3:
                    print("Current shelf surface requires at least three points.")
                    continue
                if current_points:
                    saved_polygons.setdefault(shelf_id, []).append(list(current_points))
                current_points.clear()
                if not saved_polygons.get(shelf_id):
                    print("Current shelf requires at least one completed surface.")
                    continue
                shelf_index = (shelf_index + 1) % len(shelf_ids)
            elif key == ord("s"):
                shelf_id = shelf_ids[shelf_index]
                if current_points and len(current_points) < 3:
                    print("Current shelf surface requires at least three points.")
                    continue
                if current_points:
                    saved_polygons.setdefault(shelf_id, []).append(list(current_points))
                    current_points.clear()
                missing = [
                    configured_shelf_id
                    for configured_shelf_id in shelf_ids
                    if not saved_polygons.get(configured_shelf_id)
                ]
                if missing:
                    print(f"Cannot save; missing polygons for shelves: {missing}")
                    continue
                if depth_queue is None or queue is None or intrinsics is None or calibration_id is None:
                    print(
                        "Cannot save a 3D shelf calibration from --image. "
                        "Use the live camera so synchronized aligned depth is available."
                    )
                    continue
                height, width = base_frame.shape[:2]
                try:
                    depth_frames, base_frame = _collect_synchronized_depth_frames(
                        rgb_queue=queue,
                        depth_queue=depth_queue,
                        frame_count=args.calibration_frames,
                        fallback_frame=base_frame,
                    )
                except KeyboardInterrupt:
                    print("Shelf depth capture cancelled; nothing saved.")
                    break
                calibrated_at = time.time_ns() // 1_000_000
                normalized_polygons = {
                    configured_shelf_id: tuple(
                        tuple(
                            NormalizedPoint(x / width, y / height)
                            for x, y in polygon
                        )
                        for polygon in saved_polygons[configured_shelf_id]
                    )
                    for configured_shelf_id in shelf_ids
                }
                regions = tuple(
                    ShelfRegion(
                        shelf_id=shelf_id,
                        polygon=normalized_polygons[shelf_id][0],
                        depth_tolerance_mm=args.depth_tolerance_mm,
                        depth_model=build_shelf_depth_model(
                            depth_frames,
                            polygons=normalized_polygons[shelf_id],
                            columns=args.grid_columns,
                            rows=args.grid_rows,
                            minimum_valid_fraction=args.minimum_valid_fraction,
                            minimum_valid_samples=args.minimum_valid_samples,
                            minimum_depth_mm=args.minimum_depth_mm,
                            maximum_depth_mm=args.maximum_depth_mm,
                            calibrated_at_unix_milliseconds=calibrated_at,
                            intrinsics=(
                                intrinsics.fx,
                                intrinsics.fy,
                                intrinsics.cx,
                                intrinsics.cy,
                            ),
                            camera_calibration_id=calibration_id,
                        ),
                        off_shelf_hysteresis_mm=args.off_shelf_hysteresis_mm,
                        additional_polygons=normalized_polygons[shelf_id][1:],
                    )
                    for shelf_id in shelf_ids
                )
                screenshot_dir = args.screenshot_dir or (
                    args.shelf_calibrations_root / "region_previews"
                )
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_prefix = (
                    screenshot_dir
                    / f"shelf_regions_{args.device_id}_{time.strftime('%Y%m%d_%H%M%S')}"
                )
                rgb_screenshot_path = screenshot_prefix.with_name(
                    f"{screenshot_prefix.name}_calibration_rgb.png"
                )
                preview_screenshot_path = screenshot_prefix.with_name(
                    f"{screenshot_prefix.name}_calibration_preview.png"
                )
                if not cv2.imwrite(str(rgb_screenshot_path), base_frame):
                    raise RuntimeError(
                        f"Could not write calibration RGB screenshot: {rgb_screenshot_path}"
                    )
                print(f"Saved calibration RGB screenshot to {rgb_screenshot_path}")
                if not _preview_depth_models(
                    base_frame,
                    regions=regions,
                    screenshot_path=preview_screenshot_path,
                ):
                    print("3D shelf calibration rejected; nothing saved.")
                    break
                path = shelf_regions_path(args.shelf_calibrations_root, args.device_id)
                save_camera_shelf_regions(
                    path,
                    CameraShelfRegions(
                        device_id=args.device_id,
                        regions=regions,
                        schema_version=SHELF_REGION_SCHEMA_VERSION,
                    ),
                )
                print(
                    f"Saved {len(regions)} 3D shelf regions for {args.device_id} "
                    f"using {len(depth_frames)} depth frames to {path}"
                )
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
