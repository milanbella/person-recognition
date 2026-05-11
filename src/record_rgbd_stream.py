import argparse
import json
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from pipeline.camera import (
    add_device_args,
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import DEFAULT_CAMERA_FPS, PREVIEW_HEIGHT, PREVIEW_WIDTH
from pipeline.depth import colorize_depth, intrinsics_from_matrix
from pipeline.rgbd_recording import build_recording_dir, build_recording_paths


@dataclass
class DepthPacket:
    sequence_num: int
    host_synced_seconds: float
    device_monotonic_seconds: float
    frame_mm: np.ndarray


DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record RGB video plus aligned depth frames for later depth-based replay."
    )
    add_device_args(parser)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera output FPS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help="Directory where RGBD recording folders will be written.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Optional recording duration in seconds. 0 means record until q or Ctrl+C.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show RGB and aligned depth preview while recording.",
    )
    return parser


def iso_utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def choose_best_depth_packet(
    packets: deque[DepthPacket],
    rgb_host_synced_seconds: float,
) -> DepthPacket | None:
    if not packets:
        return None
    return min(
        packets,
        key=lambda packet: abs(packet.host_synced_seconds - rgb_host_synced_seconds),
    )


def main() -> None:
    args = build_argparser().parse_args()
    device = open_or_list_devices(args)
    if device is None:
        return
    configure_live_device(device)
    print_connected_device(device)
    calibration = device.readCalibration()
    rgb_intrinsics = intrinsics_from_matrix(
        calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT),
        )
    )

    recording_dir = build_recording_dir(args.output_dir, device.getDeviceId())
    if recording_dir.exists():
        shutil.rmtree(recording_dir)
    paths = build_recording_paths(recording_dir)
    recording_dir = paths["recording_dir"]
    rgb_video_path = paths["rgb_video_path"]
    frames_path = paths["frames_path"]
    depth_frames_dir = paths["depth_frames_dir"]
    depth_frames_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(rgb_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(args.fps),
        (PREVIEW_WIDTH, PREVIEW_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open RGB video writer for {rgb_video_path}")

    frame_count = 0
    started_at = datetime.now()
    metadata_handle = frames_path.open("w", encoding="utf-8")
    metadata_handle.write(
        json.dumps(
            {
                "type": "recording_header",
                "schema_version": "2026-05-10",
                "device_id": str(device.getDeviceId()),
                "recording_started_utc": iso_utc_now(),
                "recording_dir": str(recording_dir.resolve()),
                "rgb_video_path": str(rgb_video_path.resolve()),
                "fps": args.fps,
                "width": PREVIEW_WIDTH,
                "height": PREVIEW_HEIGHT,
                "rgb_intrinsics": {
                    "fx": rgb_intrinsics.fx,
                    "fy": rgb_intrinsics.fy,
                    "cx": rgb_intrinsics.cx,
                    "cy": rgb_intrinsics.cy,
                },
            }
        )
        + "\n"
    )

    recent_depth_packets: deque[DepthPacket] = deque(maxlen=32)

    try:
        with dai.Pipeline(device) as pipeline:
            print(f"Recording RGBD stream to {recording_dir}")

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
            depth_queue = stereo.depth.createOutputQueue(maxSize=8, blocking=False)

            print("Pipeline created. Starting...")
            pipeline.start()

            while pipeline.isRunning() and not device.isClosed():
                depth_msg = depth_queue.tryGet()
                while depth_msg is not None:
                    recent_depth_packets.append(
                        DepthPacket(
                            sequence_num=int(depth_msg.getSequenceNum()),
                            host_synced_seconds=float(depth_msg.getTimestamp().total_seconds()),
                            device_monotonic_seconds=float(
                                depth_msg.getTimestampDevice().total_seconds()
                            ),
                            frame_mm=depth_msg.getFrame().copy(),
                        )
                    )
                    depth_msg = depth_queue.tryGet()

                rgb_msg = wait_for_next_frame(rgb_queue, device)
                if rgb_msg is None:
                    print("RGB stream stopped. Exiting...")
                    break

                rgb_host_synced_seconds = float(rgb_msg.getTimestamp().total_seconds())
                best_depth = choose_best_depth_packet(recent_depth_packets, rgb_host_synced_seconds)
                if best_depth is None:
                    print("RGB frame received before any aligned depth frame. Skipping frame.")
                    continue

                rgb_frame = rgb_msg.getCvFrame()
                writer.write(rgb_frame)

                depth_png_path = depth_frames_dir / f"depth_{frame_count:06d}.png"
                if not cv2.imwrite(str(depth_png_path), best_depth.frame_mm):
                    raise RuntimeError(f"Failed to write depth PNG: {depth_png_path}")

                metadata_handle.write(
                    json.dumps(
                        {
                            "type": "frame",
                            "frame_index": frame_count,
                            "rgb_sequence_num": int(rgb_msg.getSequenceNum()),
                            "rgb_host_synced_seconds": rgb_host_synced_seconds,
                            "rgb_device_monotonic_seconds": float(
                                rgb_msg.getTimestampDevice().total_seconds()
                            ),
                            "depth_sequence_num": best_depth.sequence_num,
                            "depth_host_synced_seconds": best_depth.host_synced_seconds,
                            "depth_device_monotonic_seconds": best_depth.device_monotonic_seconds,
                            "matched_depth_delta_ms": (
                                best_depth.host_synced_seconds - rgb_host_synced_seconds
                            )
                            * 1000.0,
                            "received_utc": iso_utc_now(),
                            "depth_png_relpath": str(depth_png_path.relative_to(recording_dir)),
                        }
                    )
                    + "\n"
                )
                metadata_handle.flush()
                frame_count += 1

                if args.show_preview:
                    overlay = rgb_frame.copy()
                    cv2.putText(
                        overlay,
                        f"REC RGBD {frame_count} frames",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("RGBD Recorder RGB", overlay)
                    cv2.imshow("RGBD Recorder Depth", colorize_depth(best_depth.frame_mm))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Stopping on user request.")
                        break

                if args.duration_seconds > 0:
                    elapsed = (datetime.now() - started_at).total_seconds()
                    if elapsed >= args.duration_seconds:
                        print(f"Stopping after {elapsed:.1f}s.")
                        break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except TimeoutError as exc:
        print(f"Camera stream stopped: {exc}")
    finally:
        metadata_handle.close()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved RGBD recording with {frame_count} frames to {recording_dir}")


if __name__ == "__main__":
    main()
