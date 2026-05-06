import argparse
from typing import Dict

import cv2
import depthai as dai

from pipeline.config import PREVIEW_HEIGHT, PREVIEW_WIDTH
from pipeline.detection import ScrfdInsightFaceDetector
from pipeline.entrance import (
    EntranceState,
    build_entrance_argparser,
    draw_entry_events,
    draw_entrance_debug,
    draw_entrance_line,
    process_entrance_logic,
)
from pipeline.tracking import SimpleIoUTracker, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    return build_entrance_argparser(
        description="Step 6: host-side entrance-line logic on top of SCRFD tracking."
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

    states: Dict[int, EntranceState] = {}

    with dai.Pipeline(device) as pipeline:
        print("Step 6: host-side entrance-line logic on top of SCRFD tracking.")

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
            entered_track_ids = process_entrance_logic(
                tracks=tracks,
                states=states,
                axis=args.line_axis,
                line_position=args.line_position,
                frame_shape=frame.shape[:2],
                outside_side=args.outside_side,
                min_history=args.min_history,
                debug_entrance=args.debug_entrance,
            )

            for track_id in entered_track_ids:
                print(f"ENTRY_EVENT track_id={track_id}")

            draw_tracks(frame, tracks)
            draw_entrance_line(
                frame,
                axis=args.line_axis,
                line_position=args.line_position,
                outside_side=args.outside_side,
            )
            draw_entry_events(frame, entered_track_ids)
            if args.debug_entrance:
                draw_entrance_debug(
                    frame,
                    tracks=tracks,
                    states=states,
                    axis=args.line_axis,
                    line_position=args.line_position,
                    outside_side=args.outside_side,
                )

            cv2.imshow("OAK Host SCRFD Entrance Line", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
