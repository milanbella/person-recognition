from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import depthai as dai

from pipeline.aruco_markers import (
    ArucoMarkerDetection,
    detect_aruco_markers,
    draw_rejected_aruco_candidates,
)
from pipeline.camera import (
    add_device_args,
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import DEFAULT_CAMERA_FPS, DEFAULT_MAX_RGB_DEPTH_DELTA_MS
from pipeline.depth import CameraIntrinsics, intrinsics_from_matrix
from pipeline.rgbd_recording import (
    DEFAULT_RGBD_RECORDINGS_DIR,
    RGBDReplayStream,
    load_rgbd_recording,
    resolve_recording_dir,
)
from pipeline.shelf_anchors import (
    ShelfAnchorManager,
    configured_shelf_marker_detections,
)
from pipeline.shelf_config import (
    DEFAULT_SHELF_CALIBRATIONS_DIR,
    DEFAULT_SHELF_CONFIG_PATH,
    ShelfDefinition,
    ShelfWatchingConfig,
    load_shelf_config,
)


DEFAULT_CALIBRATION_WIDTH = 1280
DEFAULT_CALIBRATION_HEIGHT = 720
CALIBRATION_CONTINUE = "continue"
CALIBRATION_SAVE = "save"
CALIBRATION_QUIT_WITHOUT_SAVE = "quit_without_save"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate stable per-camera shelf ArUco anchors from a live OAK "
            "RGB-D stream (default) or a recorded synchronized RGB-D stream."
        )
    )
    add_device_args(parser)
    parser.add_argument(
        "--source",
        choices=("live", "recording"),
        default="live",
        help="Calibration frame source. Default: live.",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help=(
            "Explicit RGB-D recording directory. Supplying this option selects "
            "recording mode even when --source is omitted."
        ),
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=DEFAULT_RGBD_RECORDINGS_DIR,
        help="Root containing oak_<device-id>.rgbd recording directories.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CALIBRATION_WIDTH,
        help="Live aligned RGB/depth width. Default: 1280.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CALIBRATION_HEIGHT,
        help="Live aligned RGB/depth height. Default: 720.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Live camera output FPS.",
    )
    parser.add_argument(
        "--depth-median-filter",
        choices=("off", "3x3", "5x5", "7x7"),
        default="7x7",
        help="StereoDepth on-device median filter. Default: 7x7.",
    )
    parser.add_argument(
        "--max-rgb-depth-delta-ms",
        type=float,
        default=DEFAULT_MAX_RGB_DEPTH_DELTA_MS,
        help="Reject live RGB/depth pairs farther apart than this. Default: 250.",
    )
    parser.add_argument(
        "--shelf-config",
        type=Path,
        default=DEFAULT_SHELF_CONFIG_PATH,
        help="Shelf catalog. Default: config/shelves.json.",
    )
    parser.add_argument(
        "--shelf-calibrations-root",
        type=Path,
        default=DEFAULT_SHELF_CALIBRATIONS_DIR,
        help="Output directory for shelf_anchors_<device-id>.json.",
    )
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-spread-mm", type=float, default=50.0)
    parser.add_argument("--movement-tolerance-mm", type=float, default=150.0)
    parser.add_argument("--marker-min-valid-pixels", type=int, default=5)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth source frame. Default: 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum processed frames; zero means run until q or Ctrl+C.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show detected shelf markers while processing.",
    )
    parser.add_argument(
        "--show-rejected-candidates",
        action="store_true",
        help=(
            "Draw quadrilaterals that resemble markers but failed decoding. "
            "Useful for diagnosing print quality, blur, and dictionary mismatch."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Save valid anchors without an interactive confirmation.",
    )
    return parser


def calibration_source(args: argparse.Namespace) -> str:
    if args.recording_dir is not None:
        return "recording"
    return str(args.source)


def calibration_action_for_key(key: int, *, valid_anchor_count: int) -> str:
    if key == ord("q"):
        return CALIBRATION_QUIT_WITHOUT_SAVE
    if key == ord("s") and valid_anchor_count > 0:
        return CALIBRATION_SAVE
    return CALIBRATION_CONTINUE


def _valid_candidate_count(
    manager: ShelfAnchorManager,
    shelves: tuple[ShelfDefinition, ...],
) -> int:
    return sum(
        1
        for shelf in shelves
        for marker_id in shelf.all_marker_ids
        if (
            (
                candidate := manager.candidate_anchor(
                    shelf.shelf_id,
                    marker_id,
                )
            )
            is not None
            and candidate.sample_count >= manager.min_samples
            and candidate.rms_spread_mm <= manager.max_spread_mm
        )
    )


def _draw_detections(
    frame,
    detections: list[ArucoMarkerDetection],
    shelves_by_marker: dict[int, ShelfDefinition],
    valid_depth_marker_ids: set[int],
) -> None:
    for detection in detections:
        shelf = shelves_by_marker.get(detection.marker_id)
        if shelf is None:
            color = (0, 165, 255)
            label = f"marker={detection.marker_id} UNCONFIGURED"
        elif detection.marker_id not in valid_depth_marker_ids:
            color = (0, 255, 255)
            label = (
                f"shelf={shelf.shelf_id} marker={detection.marker_id} "
                "DEPTH INVALID"
            )
        else:
            color = (0, 220, 0)
            label = (
                f"shelf={shelf.shelf_id} marker={detection.marker_id} {shelf.label}"
            )
        corners = [(int(round(x)), int(round(y))) for x, y in detection.corners_px]
        for start, end in zip(corners, corners[1:] + corners[:1]):
            cv2.line(frame, start, end, color, 2)
        center = (
            int(round(detection.center_px[0])),
            int(round(detection.center_px[1])),
        )
        cv2.circle(frame, center, 5, color, -1)
        cv2.putText(
            frame,
            label,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _create_manager(
    args: argparse.Namespace,
    *,
    device_id: str,
    config: ShelfWatchingConfig,
) -> tuple[ShelfAnchorManager, tuple[ShelfDefinition, ...]]:
    shelves = config.shelves
    return (
        ShelfAnchorManager(
            device_id=device_id,
            config=config,
            calibration_root=args.shelf_calibrations_root,
            min_samples=args.min_samples,
            max_spread_mm=args.max_spread_mm,
            movement_tolerance_mm=args.movement_tolerance_mm,
            marker_min_valid_pixels=args.marker_min_valid_pixels,
            auto_save=False,
            load_existing=False,
        ),
        shelves,
    )


def _process_frame(
    *,
    rgb_frame,
    depth_frame,
    host_synced_seconds: float,
    intrinsics: CameraIntrinsics,
    config: ShelfWatchingConfig,
    shelves: tuple[ShelfDefinition, ...],
    manager: ShelfAnchorManager,
    show_preview: bool,
    show_rejected_candidates: bool,
) -> str:
    result = detect_aruco_markers(
        rgb_frame,
        dictionary_name=config.aruco_dictionary,
    )
    shelf_detections = configured_shelf_marker_detections(
        result.detections,
        shelves,
    )
    observations = manager.process_detections(
        shelf_detections,
        depth_frame_mm=depth_frame,
        intrinsics=intrinsics,
        host_synced_seconds=host_synced_seconds,
        observed_at_unix_milliseconds=time.time_ns() // 1_000_000,
        log_trace=False,
    )
    if not show_preview:
        return CALIBRATION_CONTINUE

    preview = rgb_frame.copy()
    _draw_detections(
        preview,
        result.detections,
        {
            marker_id: shelf
            for shelf in shelves
            for marker_id in shelf.all_marker_ids
        },
        {observation.marker_id for observation in observations},
    )
    if show_rejected_candidates:
        draw_rejected_aruco_candidates(preview, result.rejected_candidates)
    detected_ids = sorted(detection.marker_id for detection in result.detections)
    status_color = (0, 220, 0) if detected_ids else (0, 0, 255)
    cv2.putText(
        preview,
        (
            f"dictionary={config.aruco_dictionary} detected={detected_ids} "
            f"catalog_markers={sum(len(shelf.all_marker_ids) for shelf in shelves)} "
            f"rejected={len(result.rejected_candidates)}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        (
            "green=usable yellow=bad-depth orange=unconfigured  "
            f"ready={_valid_candidate_count(manager, shelves)}  "
            "s=save q=quit-without-save"
        ),
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Shelf Anchor Calibration", preview)
    key = cv2.waitKey(1) & 0xFF
    action = calibration_action_for_key(
        key,
        valid_anchor_count=_valid_candidate_count(manager, shelves),
    )
    if key == ord("s") and action == CALIBRATION_CONTINUE:
        print(
            "Save requested, but no shelf anchor has enough stable samples yet. "
            "Calibration will continue."
        )
    return action


def _drain_queue(queue, messages: deque) -> None:
    message = queue.tryGet()
    while message is not None:
        messages.append(message)
        message = queue.tryGet()


def _latest_rgb_message(rgb_queue, device: dai.Device):
    message = wait_for_next_frame(rgb_queue, device)
    if message is None:
        return None
    newer = rgb_queue.tryGet()
    while newer is not None:
        message = newer
        newer = rgb_queue.tryGet()
    return message


def _matching_depth_message(
    *,
    depth_queue,
    recent_depth_messages: deque,
    rgb_host_synced_seconds: float,
    device: dai.Device,
    wait_seconds: float = 1.0,
):
    deadline = time.monotonic() + wait_seconds
    while True:
        _drain_queue(depth_queue, recent_depth_messages)
        if recent_depth_messages:
            newest_seconds = float(
                recent_depth_messages[-1].getTimestamp().total_seconds()
            )
            if newest_seconds >= rgb_host_synced_seconds:
                break
        if (
            time.monotonic() >= deadline
            or device.isClosed()
            or not device.isPipelineRunning()
        ):
            break
        time.sleep(0.005)

    if not recent_depth_messages:
        return None
    return min(
        recent_depth_messages,
        key=lambda message: abs(
            float(message.getTimestamp().total_seconds())
            - rgb_host_synced_seconds
        ),
    )


def _run_live(
    args: argparse.Namespace,
    config: ShelfWatchingConfig,
) -> tuple[ShelfAnchorManager, tuple[ShelfDefinition, ...], int, str] | None:
    device = open_or_list_devices(args)
    if device is None:
        return None
    configure_live_device(device)
    print_connected_device(device)
    device_id = str(device.getDeviceId())
    calibration = device.readCalibration()
    intrinsics = intrinsics_from_matrix(
        calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            (args.width, args.height),
        )
    )
    manager, shelves = _create_manager(
        args,
        device_id=device_id,
        config=config,
    )
    recent_depth_messages: deque = deque(maxlen=32)
    seen_frames = 0
    processed = 0
    rejected_pairs = 0
    exit_action = CALIBRATION_CONTINUE

    try:
        with dai.Pipeline(device) as pipeline:
            cam_rgb = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A
            )
            rgb_output = cam_rgb.requestOutput(
                size=(args.width, args.height),
                type=dai.ImgFrame.Type.BGR888p,
                fps=args.fps,
            )
            mono_left = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B
            )
            mono_right = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C
            )
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setMedianFilter(
                {
                    "off": dai.MedianFilter.MEDIAN_OFF,
                    "3x3": dai.MedianFilter.KERNEL_3x3,
                    "5x5": dai.MedianFilter.KERNEL_5x5,
                    "7x7": dai.MedianFilter.KERNEL_7x7,
                }[args.depth_median_filter]
            )
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(args.width, args.height)
            mono_left.requestFullResolutionOutput(fps=args.fps).link(stereo.left)
            mono_right.requestFullResolutionOutput(fps=args.fps).link(stereo.right)

            rgb_queue = rgb_output.createOutputQueue(maxSize=2, blocking=False)
            depth_queue = stereo.depth.createOutputQueue(maxSize=8, blocking=False)
            pipeline.start()
            print(
                f"Calibrating live shelf anchors device_id={device_id} "
                f"size={args.width}x{args.height}. Press s to save or q to "
                "quit without saving."
            )

            while pipeline.isRunning() and not device.isClosed():
                rgb_message = _latest_rgb_message(rgb_queue, device)
                if rgb_message is None:
                    break
                seen_frames += 1
                if (seen_frames - 1) % args.frame_step != 0:
                    continue

                rgb_seconds = float(rgb_message.getTimestamp().total_seconds())
                depth_message = _matching_depth_message(
                    depth_queue=depth_queue,
                    recent_depth_messages=recent_depth_messages,
                    rgb_host_synced_seconds=rgb_seconds,
                    device=device,
                )
                if depth_message is None:
                    rejected_pairs += 1
                    continue
                depth_seconds = float(depth_message.getTimestamp().total_seconds())
                delta_ms = abs(depth_seconds - rgb_seconds) * 1000.0
                if delta_ms > args.max_rgb_depth_delta_ms:
                    rejected_pairs += 1
                    continue

                action = _process_frame(
                    rgb_frame=rgb_message.getCvFrame(),
                    depth_frame=depth_message.getFrame(),
                    host_synced_seconds=rgb_seconds,
                    intrinsics=intrinsics,
                    config=config,
                    shelves=shelves,
                    manager=manager,
                    show_preview=args.show_preview,
                    show_rejected_candidates=args.show_rejected_candidates,
                )
                processed += 1
                if action != CALIBRATION_CONTINUE:
                    exit_action = action
                    break
                if args.max_frames and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        exit_action = CALIBRATION_QUIT_WITHOUT_SAVE
        print("Shelf calibration interrupted; nothing will be saved.")
    finally:
        if args.show_preview:
            cv2.destroyAllWindows()
        device.close()

    if rejected_pairs:
        print(
            f"Rejected live RGB/depth pairs: {rejected_pairs} "
            f"(maximum delta {args.max_rgb_depth_delta_ms:.1f} ms)"
        )
    return manager, shelves, processed, exit_action


def _run_recording(
    args: argparse.Namespace,
    config: ShelfWatchingConfig,
) -> tuple[ShelfAnchorManager, tuple[ShelfDefinition, ...], int, str]:
    recording_dir = resolve_recording_dir(
        recording_dir=args.recording_dir,
        device_id=args.device_id,
        recordings_root=args.recordings_root,
    )
    recording = load_rgbd_recording(recording_dir)
    if recording.rgb_intrinsics is None:
        raise RuntimeError(
            "The recording has no RGB intrinsics. Re-record it with the "
            "current record_rgbd_stream.py."
        )
    intrinsics = CameraIntrinsics(
        fx=float(recording.rgb_intrinsics["fx"]),
        fy=float(recording.rgb_intrinsics["fy"]),
        cx=float(recording.rgb_intrinsics["cx"]),
        cy=float(recording.rgb_intrinsics["cy"]),
    )
    manager, shelves = _create_manager(
        args,
        device_id=recording.device_id,
        config=config,
    )
    replay = RGBDReplayStream(recording)
    processed = 0
    exit_action = CALIBRATION_CONTINUE
    try:
        while replay.current_frame_meta is not None:
            frame_index = replay.current_index
            rgb_frame = replay.current_rgb_frame
            depth_frame = replay.current_depth_frame
            frame_meta = replay.current_frame_meta
            if (
                frame_index % args.frame_step == 0
                and rgb_frame is not None
                and depth_frame is not None
            ):
                action = _process_frame(
                    rgb_frame=rgb_frame,
                    depth_frame=depth_frame,
                    host_synced_seconds=frame_meta.rgb_host_synced_seconds,
                    intrinsics=intrinsics,
                    config=config,
                    shelves=shelves,
                    manager=manager,
                    show_preview=args.show_preview,
                    show_rejected_candidates=args.show_rejected_candidates,
                )
                processed += 1
                if action != CALIBRATION_CONTINUE:
                    exit_action = action
                    break
                if args.max_frames and processed >= args.max_frames:
                    break
            if not replay.advance():
                break
    except KeyboardInterrupt:
        exit_action = CALIBRATION_QUIT_WITHOUT_SAVE
        print("Shelf calibration interrupted; nothing will be saved.")
    finally:
        replay.close()
        if args.show_preview:
            cv2.destroyAllWindows()
    return manager, shelves, processed, exit_action


def _report_and_save(
    args: argparse.Namespace,
    *,
    manager: ShelfAnchorManager,
    shelves: tuple[ShelfDefinition, ...],
    save_requested: bool = False,
) -> None:
    valid_count = 0
    unseen_marker_ids: list[int] = []
    for shelf in shelves:
        for marker_id in shelf.all_marker_ids:
            candidate = manager.candidate_anchor(shelf.shelf_id, marker_id)
            if candidate is None:
                unseen_marker_ids.append(marker_id)
                continue
            valid = (
                candidate.sample_count >= args.min_samples
                and candidate.rms_spread_mm <= args.max_spread_mm
            )
            print(
                f"Shelf {shelf.shelf_id} marker={marker_id}: "
                f"samples={candidate.sample_count} "
                f"spread_mm={candidate.rms_spread_mm:.1f} "
                f"point_mm={tuple(round(value, 1) for value in candidate.point_3d_mm)} "
                f"status={'valid' if valid else 'rejected'}"
            )
            if valid:
                valid_count += 1
    if unseen_marker_ids:
        print(
            "Configured markers not seen by this camera (not saved): "
            f"{unseen_marker_ids}"
        )
    if valid_count == 0:
        raise RuntimeError("No shelf anchors passed the calibration quality checks.")

    should_save = save_requested or args.yes
    if not should_save:
        response = input(
            f"Save {valid_count} valid shelf anchors to "
            f"{manager.calibration_path}? [y/N] "
        )
        should_save = response.strip().lower() in {"y", "yes"}
    if should_save:
        manager.accept_candidates()
        print(f"Saved shelf anchors to {manager.calibration_path}")
    else:
        print("Shelf anchor calibration was not saved.")


def main() -> None:
    args = build_argparser().parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must not be negative.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.max_rgb_depth_delta_ms < 0:
        raise ValueError("--max-rgb-depth-delta-ms must not be negative.")

    config = load_shelf_config(args.shelf_config)
    source = calibration_source(args)
    if source == "live":
        result = _run_live(args, config)
        if result is None:
            return
    else:
        result = _run_recording(args, config)
    manager, shelves, processed, exit_action = result
    print(f"Processed {source} frames: {processed}")
    if exit_action == CALIBRATION_QUIT_WITHOUT_SAVE:
        print("Shelf anchor calibration quit without saving.")
        return
    _report_and_save(
        args,
        manager=manager,
        shelves=shelves,
        save_requested=exit_action == CALIBRATION_SAVE,
    )


if __name__ == "__main__":
    main()
