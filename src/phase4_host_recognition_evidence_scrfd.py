import argparse
from pathlib import Path
from typing import Dict

import cv2
import depthai as dai
from pipeline.camera import (
    configure_live_device,
    open_or_list_devices,
    print_connected_device,
    wait_for_next_frame,
)
from pipeline.config import (
    DEFAULT_EVIDENCE_CROP_MARGIN,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_EVIDENCE_POST_FRAMES,
    DEFAULT_EVIDENCE_PRE_FRAMES,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)
from pipeline.detection import build_person_detector
from pipeline.entrance import (
    EntranceState,
    build_entrance_argparser,
    draw_entry_events,
    draw_entrance_line,
    process_entrance_logic,
)
from pipeline.evidence import EvidenceCollector, draw_evidence_status
from pipeline.tracking import build_person_tracker, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    parser = build_entrance_argparser(
        description=(
        "Step 7: host-side recognition evidence capture on top of SCRFD entrance events."
        )
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory where evidence crops will be saved.",
    )
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=DEFAULT_EVIDENCE_PRE_FRAMES,
        help="How many recent crops to keep and save before the entry event.",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=DEFAULT_EVIDENCE_POST_FRAMES,
        help="How many crops to save after the entry event.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=DEFAULT_EVIDENCE_CROP_MARGIN,
        help="Extra crop margin as a fraction of box width/height.",
    )
    return parser

def main() -> None:
    args = build_argparser().parse_args()
    device = open_or_list_devices(args)
    if device is None:
        return
    configure_live_device(device)

    detector = build_person_detector(args)
    tracker = build_person_tracker(args)
    collector = EvidenceCollector(
        evidence_dir=args.evidence_dir,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
    )

    print_connected_device(device)

    entrance_states: Dict[int, EntranceState] = {}
    frame_index = 0

    with dai.Pipeline(device) as pipeline:
        print("Step 7: host-side recognition evidence capture on top of SCRFD entrance events.")

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

                frame_index += 1
                frame = msg.getCvFrame()

                tracks = tracker.update(detector.detect(frame))
                collector.update_buffers(
                    frame=frame,
                    tracks=tracks,
                    frame_index=frame_index,
                    crop_margin=args.crop_margin,
                )

                entered_track_ids = process_entrance_logic(
                    tracks=tracks,
                    states=entrance_states,
                    axis=args.line_axis,
                    line_position=args.line_position,
                    frame_shape=frame.shape[:2],
                    outside_side=args.outside_side,
                    min_history=args.min_history,
                    debug_entrance=args.debug_entrance,
                )

                for track_id in entered_track_ids:
                    print(f"ENTRY_EVENT track_id={track_id}")
                    collector.record_entry_event(track_id)

                draw_tracks(frame, tracks)
                draw_entrance_line(
                    frame,
                    axis=args.line_axis,
                    line_position=args.line_position,
                    outside_side=args.outside_side,
                )
                draw_entry_events(frame, entered_track_ids)
                draw_evidence_status(frame, collector)

                cv2.imshow("OAK Host SCRFD Recognition Evidence", frame)
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
