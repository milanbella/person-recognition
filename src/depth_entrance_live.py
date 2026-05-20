import argparse
from typing import Dict

import cv2
import depthai as dai

from pipeline.camera import (
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import PREVIEW_HEIGHT, PREVIEW_WIDTH
from pipeline.depth import (
    CameraIntrinsics,
    DepthEntranceState,
    build_depth_entrance_argparser,
    colorize_depth,
    draw_depth_event_banner,
    draw_depth_samples,
    intrinsics_from_matrix,
    plane_enter_direction_from_args,
    plane_from_args,
    process_depth_plane_logic,
    process_depth_entrance_logic,
)
from pipeline.detection import build_person_detector
from pipeline.tracking import SimpleIoUTracker, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    return build_depth_entrance_argparser(
        description="Depth-based entrance prototype using RGB plus stereo depth aligned to RGB."
    )


def main() -> None:
    args = build_argparser().parse_args()
    device = open_or_list_devices(args)
    if device is None:
        return
    configure_live_device(device)

    detector = build_person_detector(args)
    tracker = SimpleIoUTracker(
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
    )

    print_connected_device(device)
    print("Depth prototype uses CAM_A RGB plus CAM_B/C stereo depth aligned to RGB.")
    calibration = device.readCalibration()
    rgb_intrinsics: CameraIntrinsics = intrinsics_from_matrix(
        calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT),
        )
    )
    plane = plane_from_args(args)
    plane_enter_direction = plane_enter_direction_from_args(args)
    print(
        f"RGB intrinsics fx={rgb_intrinsics.fx:.1f} fy={rgb_intrinsics.fy:.1f} "
        f"cx={rgb_intrinsics.cx:.1f} cy={rgb_intrinsics.cy:.1f}"
    )
    if args.depth_trigger_mode == "plane" and args.plane_json is not None:
        print(f"Loaded plane from {args.plane_json}")

    depth_states: Dict[int, DepthEntranceState] = {}
    event_flash_remaining = 0
    event_flash_text = ""

    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_output = cam_rgb.requestOutput(
            size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )

        mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)

        mono_left.requestFullResolutionOutput(fps=args.fps).link(stereo.left)
        mono_right.requestFullResolutionOutput(fps=args.fps).link(stereo.right)

        rgb_queue = rgb_output.createOutputQueue(maxSize=4, blocking=False)
        depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        print("Depth entrance prototype running.")
        print("Press q to quit.")
        pipeline.start()

        latest_depth_frame = None
        latest_depth_visual = None

        try:
            while pipeline.isRunning() and not device.isClosed():
                depth_message = depth_queue.tryGet()
                while depth_message is not None:
                    latest_depth_frame = depth_message.getFrame()
                    latest_depth_visual = colorize_depth(latest_depth_frame)
                    depth_message = depth_queue.tryGet()

                rgb_message = wait_for_next_frame(rgb_queue, device)
                if rgb_message is None:
                    print("RGB stream stopped. Exiting...")
                    break

                frame = rgb_message.getCvFrame()
                if latest_depth_frame is None:
                    cv2.putText(
                        frame,
                        "Waiting for aligned depth...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Depth Entrance Prototype", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                detections = detector.detect(frame)
                tracks = tracker.update(detections)

                if args.depth_trigger_mode == "plane":
                    entered_track_ids, depth_samples, signed_distances_mm = process_depth_plane_logic(
                        tracks=tracks,
                        depth_frame_mm=latest_depth_frame,
                        intrinsics=rgb_intrinsics,
                        states=depth_states,
                        plane=plane,
                        plane_enter_direction=plane_enter_direction,
                        plane_hysteresis_mm=float(args.plane_hysteresis_mm),
                        min_valid_pixels=args.depth_min_valid_pixels,
                        roi_width_fraction=args.depth_roi_width_fraction,
                        roi_height_fraction=args.depth_roi_height_fraction,
                    )
                else:
                    entered_track_ids, depth_samples = process_depth_entrance_logic(
                        tracks=tracks,
                        depth_frame_mm=latest_depth_frame,
                        intrinsics=rgb_intrinsics,
                        states=depth_states,
                        depth_threshold_mm=float(args.depth_threshold_mm),
                        depth_hysteresis_mm=float(args.depth_hysteresis_mm),
                        min_valid_pixels=args.depth_min_valid_pixels,
                        roi_width_fraction=args.depth_roi_width_fraction,
                        roi_height_fraction=args.depth_roi_height_fraction,
                    )
                    signed_distances_mm = {}

                for track_id in entered_track_ids:
                    sample = depth_samples.get(track_id)
                    if sample is None:
                        continue
                    if args.depth_trigger_mode == "plane":
                        signed_mm = signed_distances_mm.get(track_id, float("nan"))
                        print(
                            f"DEPTH_PLANE_ENTRY_EVENT track_id={track_id} "
                            f"depth_mm={sample.depth_mm:.0f} plane_mm={signed_mm:.0f}"
                        )
                    else:
                        print(
                            f"DEPTH_ENTRY_EVENT track_id={track_id} depth_mm={sample.depth_mm:.0f}"
                        )
                if entered_track_ids:
                    prefix = "PLANE ENTRY" if args.depth_trigger_mode == "plane" else "DEPTH ENTRY"
                    event_flash_text = f"{prefix}: " + ", ".join(
                        str(track_id) for track_id in entered_track_ids
                    )
                    event_flash_remaining = 12

                draw_tracks(frame, tracks)
                draw_depth_samples(
                    frame,
                    tracks=tracks,
                    depth_samples=depth_samples,
                    depth_threshold_mm=float(args.depth_threshold_mm),
                    signed_distances_mm=signed_distances_mm,
                    plane_mode=args.depth_trigger_mode == "plane",
                )
                if event_flash_remaining > 0:
                    draw_depth_event_banner(frame, event_flash_text)
                    event_flash_remaining -= 1
                cv2.imshow("Depth Entrance Prototype", frame)

                if args.show_depth_window and latest_depth_visual is not None:
                    cv2.imshow("Aligned Depth", latest_depth_visual)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        except TimeoutError as exc:
            print(f"Camera stream stopped: {exc}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
