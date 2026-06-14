from __future__ import annotations

import argparse
import time

import cv2
import depthai as dai

from pipeline.aruco_markers import (
    DEFAULT_ARUCO_DICTIONARY,
    DEFAULT_DOOR_MARKER_IDS,
    detect_aruco_markers,
    draw_aruco_detections,
    draw_rejected_aruco_candidates,
    format_marker_summary,
)
from pipeline.camera import (
    add_device_args,
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import DEFAULT_CAMERA_FPS


DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
WINDOW_NAME = "Door ArUco Detection"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live OAK RGB prototype for detecting ArUco markers around the entrance door."
    )
    add_device_args(parser)
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Requested RGB stream width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Requested RGB stream height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera output FPS.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default=DEFAULT_ARUCO_DICTIONARY,
        help="OpenCV ArUco dictionary name, for example DICT_4X4_50.",
    )
    parser.add_argument(
        "--door-marker-id",
        type=int,
        nargs="*",
        default=list(DEFAULT_DOOR_MARKER_IDS),
        help="Marker IDs that should be highlighted as door markers.",
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Draw rejected ArUco candidate quads in red for detection debugging.",
    )
    parser.add_argument(
        "--summary-interval-seconds",
        type=float,
        default=1.0,
        help="How often to print marker summaries to the console.",
    )
    return parser


def draw_status_overlay(
    frame,
    *,
    detected_count: int,
    visible_door_count: int,
    rejected_count: int,
    width: int,
    height: int,
) -> None:
    lines = [
        f"ArUco {width}x{height} markers={detected_count} door_visible={visible_door_count}",
        "q=quit",
    ]
    if rejected_count:
        lines[0] += f" rejected={rejected_count}"

    x, y = 18, 36
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34


def main() -> None:
    args = build_argparser().parse_args()
    door_marker_ids = set(args.door_marker_id)

    device = open_or_list_devices(args)
    if device is None:
        return
    configure_live_device(device)
    print_connected_device(device)

    with dai.Pipeline(device) as pipeline:
        print(
            "Starting door ArUco detection "
            f"size={args.width}x{args.height} fps={args.fps} "
            f"dictionary={args.dictionary} door_marker_ids={sorted(door_marker_ids)}"
        )

        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(args.width, args.height),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )
        queue = camera_out.createOutputQueue(maxSize=4, blocking=False)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        pipeline.start()
        last_summary_seconds = 0.0

        try:
            while pipeline.isRunning() and not device.isClosed():
                msg = wait_for_next_frame(queue, device)
                if msg is None:
                    print("Camera stopped delivering frames. Exiting...")
                    break

                frame = msg.getCvFrame()
                result = detect_aruco_markers(frame, dictionary_name=args.dictionary)
                visible_door_count = sum(
                    1 for detection in result.detections if detection.marker_id in door_marker_ids
                )

                draw_aruco_detections(
                    frame,
                    result.detections,
                    door_marker_ids=door_marker_ids,
                )
                if args.show_rejected:
                    draw_rejected_aruco_candidates(frame, result.rejected_candidates)

                draw_status_overlay(
                    frame,
                    detected_count=len(result.detections),
                    visible_door_count=visible_door_count,
                    rejected_count=len(result.rejected_candidates) if args.show_rejected else 0,
                    width=args.width,
                    height=args.height,
                )

                now_seconds = time.monotonic()
                if now_seconds - last_summary_seconds >= args.summary_interval_seconds:
                    print(
                        format_marker_summary(
                            result.detections,
                            door_marker_ids=door_marker_ids,
                        )
                    )
                    last_summary_seconds = now_seconds

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Exiting...")
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        except TimeoutError as exc:
            print(f"Camera stream stopped: {exc}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
