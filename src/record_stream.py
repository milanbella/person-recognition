import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

from pipeline.camera import (
    add_device_args,
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import DEFAULT_CAMERA_FPS


PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a full RGB stream from one OAK camera to a video file."
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
        help="Directory where recorded videos will be written.",
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
        help="Show a live preview while recording.",
    )
    return parser


def build_output_path(output_dir: Path, device_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"oak_{device_id}.avi"


def iso_utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def main() -> None:
    args = build_argparser().parse_args()
    device = open_or_list_devices(args)
    if device is None:
        return
    configure_live_device(device)
    print_connected_device(device)

    output_path = build_output_path(args.output_dir, device.getDeviceId())
    metadata_path = output_path.with_suffix(".timestamps.jsonl")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(args.fps),
        (PREVIEW_WIDTH, PREVIEW_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    frame_count = 0
    started_at = datetime.now()
    metadata_handle = metadata_path.open("w", encoding="utf-8")
    metadata_handle.write(
        json.dumps(
            {
                "type": "recording_header",
                "schema_version": "2026-05-10",
                "device_id": str(device.getDeviceId()),
                "recording_started_utc": iso_utc_now(),
                "video_path": str(output_path.resolve()),
                "fps": args.fps,
                "width": PREVIEW_WIDTH,
                "height": PREVIEW_HEIGHT,
            }
        )
        + "\n"
    )

    try:
        with dai.Pipeline(device) as pipeline:
            print(f"Recording stream to {output_path}")
            print(f"Writing timestamps to {metadata_path}")

            camera = pipeline.create(dai.node.Camera).build()
            camera_out = camera.requestOutput(
                size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
                type=dai.ImgFrame.Type.BGR888p,
                fps=args.fps,
            )
            queue = camera_out.createOutputQueue(maxSize=4, blocking=False)

            print("Pipeline created. Starting...")
            pipeline.start()

            while pipeline.isRunning() and not device.isClosed():
                msg = wait_for_next_frame(queue, device)
                if msg is None:
                    print("Camera stopped delivering frames. Exiting...")
                    break

                frame = msg.getCvFrame()
                writer.write(frame)
                metadata_handle.write(
                    json.dumps(
                        {
                            "type": "frame",
                            "frame_index": frame_count,
                            "sequence_num": int(msg.getSequenceNum()),
                            "host_synced_seconds": float(msg.getTimestamp().total_seconds()),
                            "device_monotonic_seconds": float(
                                msg.getTimestampDevice().total_seconds()
                            ),
                            "received_utc": iso_utc_now(),
                        }
                    )
                    + "\n"
                )
                frame_count += 1

                if args.show_preview:
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        f"REC {frame_count} frames",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("OAK Stream Recorder", overlay)
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

    print(f"Saved {frame_count} frames to {output_path}")
    print(f"Saved frame timestamps to {metadata_path}")


if __name__ == "__main__":
    main()
