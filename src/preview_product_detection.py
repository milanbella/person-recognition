from __future__ import annotations

import argparse
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
from pipeline.product_detection import (
    DEFAULT_PRODUCT_MODEL,
    DEFAULT_PRODUCT_SCORE_THRESHOLD,
    YoloOnnxProductDetector,
    draw_product_detections,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview custom YOLO product recognition on one OAK camera."
    )
    add_device_args(parser)
    parser.add_argument("--model", type=Path, default=DEFAULT_PRODUCT_MODEL)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=min(DEFAULT_CAMERA_FPS, 15))
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_PRODUCT_SCORE_THRESHOLD,
    )
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    return parser


def _latest_message(queue, device: dai.Device):
    message = wait_for_next_frame(queue, device)
    if message is None:
        return None
    newer = queue.tryGet()
    while newer is not None:
        message = newer
        newer = queue.tryGet()
    return message


def main() -> None:
    args = build_argparser().parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Camera width, height and FPS must be positive.")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between zero and one.")

    device = open_or_list_devices(args)
    if device is None:
        return
    configure_live_device(device)
    print_connected_device(device)
    detector = YoloOnnxProductDetector(
        args.model,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )

    with dai.Pipeline(device) as pipeline:
        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(args.width, args.height),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )
        queue = camera_out.createOutputQueue(maxSize=2, blocking=False)
        pipeline.start()
        cv2.namedWindow("Product Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "Product Detection", args.display_width, args.display_height
        )
        previous_summary: tuple[tuple[str, int], ...] = ()
        print("Product preview started. Press q to quit.")
        try:
            while pipeline.isRunning() and not device.isClosed():
                message = _latest_message(queue, device)
                if message is None:
                    break
                frame = message.getCvFrame()
                detections = detector.detect(frame)
                counts: dict[str, int] = {}
                for detection in detections:
                    counts[detection.label] = counts.get(detection.label, 0) + 1
                summary = tuple(sorted(counts.items()))
                if summary != previous_summary:
                    print(f"Products: {dict(summary)}")
                    previous_summary = summary
                preview = frame.copy()
                draw_product_detections(preview, detections)
                cv2.imshow("Product Detection", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Product preview interrupted.")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
