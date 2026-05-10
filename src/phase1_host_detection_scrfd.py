import argparse

import cv2
import depthai as dai
from pipeline.camera import open_or_list_devices, print_connected_device
from pipeline.config import PREVIEW_HEIGHT, PREVIEW_WIDTH
from pipeline.detection import (
    ScrfdInsightFaceDetector,
    build_detection_argparser,
    draw_detections,
)


def build_argparser() -> argparse.ArgumentParser:
    return build_detection_argparser(
        description="Step 2: host-side SCRFD detection on OAK USB frames."
    )


def main() -> None:
    args = build_argparser().parse_args()
    device = open_or_list_devices(args)
    if device is None:
        return

    detector = ScrfdInsightFaceDetector(
        model_path=args.model,
        input_size=(args.input_width, args.input_height),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )

    print_connected_device(device)

    with dai.Pipeline(device) as pipeline:
        print("Step 2: host-side SCRFD detection on OAK USB frames.")

        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )
        queue = camera_out.createOutputQueue(maxSize=4, blocking=False)

        print("Pipeline created. Starting...")
        pipeline.start()

        while pipeline.isRunning():
            msg = queue.get()
            frame = msg.getCvFrame()

            detections = detector.detect(frame)
            draw_detections(frame, detections)

            cv2.imshow("OAK Host SCRFD Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
