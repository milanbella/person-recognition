import argparse

import cv2
import depthai as dai
from pipeline.camera import (
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import PREVIEW_HEIGHT, PREVIEW_WIDTH
from pipeline.detection import (
    build_detection_argparser,
    build_person_detector,
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
    configure_live_device(device)

    detector = build_person_detector(args)

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

        try:
            while pipeline.isRunning() and not device.isClosed():
                msg = wait_for_next_frame(queue, device)
                if msg is None:
                    print("Camera stopped delivering frames. Exiting...")
                    break

                frame = msg.getCvFrame()

                detections = detector.detect(frame)
                draw_detections(frame, detections)

                cv2.imshow("OAK Host SCRFD Detection", frame)

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
