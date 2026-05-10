import argparse

import cv2
import depthai as dai
from pipeline.camera import add_device_args, open_or_list_devices, print_connected_device
from pipeline.config import DEFAULT_CAMERA_FPS


PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1: host-side RGB frame capture and preview."
    )
    add_device_args(parser)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera output FPS.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = open_or_list_devices(args)
    if device is None:
        return
    print_connected_device(device)

    with dai.Pipeline(device) as pipeline:
        print("Step 1: host-side RGB frame capture and preview.")

        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )
        queue = camera_out.createOutputQueue(
            maxSize=4,
            blocking=False,
        )

        print("Pipeline created. Starting...")
        pipeline.start()

        while pipeline.isRunning():
            msg = queue.get()
            frame = msg.getCvFrame()

            cv2.imshow("OAK RGB Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
