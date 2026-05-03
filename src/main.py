import cv2
import depthai as dai


PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720


def main() -> None:
    device = dai.Device()
    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()} Platform: {platform}")

    with dai.Pipeline(device) as pipeline:
        print("Step 1: host-side RGB frame capture and preview.")

        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=30,
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
