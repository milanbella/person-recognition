import argparse

import cv2
import depthai as dai
from pipeline.config import PREVIEW_HEIGHT, PREVIEW_WIDTH
from pipeline.detection import ScrfdInsightFaceDetector
from pipeline.tracking import SimpleIoUTracker, build_tracking_argparser, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    return build_tracking_argparser(
        description="Step 3/4: host-side tracking on top of host-side SCRFD detections."
    )

def main() -> None:
    args = build_argparser().parse_args()

    detector = ScrfdInsightFaceDetector(
        model_path=args.model,
        input_size=(args.input_width, args.input_height),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    tracker = SimpleIoUTracker(
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
    )

    device = dai.Device()
    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()} Platform: {platform}")

    with dai.Pipeline(device) as pipeline:
        print("Step 3/4: host-side tracking on top of host-side SCRFD detections.")

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
            tracks = tracker.update(detections)
            draw_tracks(frame, tracks)

            cv2.imshow("OAK Host SCRFD Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
