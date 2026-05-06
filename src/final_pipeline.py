import argparse
from pathlib import Path
from typing import Dict

import cv2
import depthai as dai

from pipeline.config import (
    DEFAULT_EVIDENCE_CROP_MARGIN,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_EVIDENCE_POST_FRAMES,
    DEFAULT_EVIDENCE_PRE_FRAMES,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)
from pipeline.detection import ScrfdInsightFaceDetector
from pipeline.entrance import (
    EntranceState,
    build_entrance_argparser,
    draw_entry_events,
    draw_entrance_debug,
    draw_entrance_line,
    process_entrance_logic,
)
from pipeline.evidence import EvidenceCollector, draw_evidence_status
from pipeline.tracking import SimpleIoUTracker, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    parser = build_entrance_argparser(
        description="Unified live pipeline: detection, tracking, entrance events, and evidence capture."
    )
    parser.add_argument(
        "--write-evidence",
        action="store_true",
        help="Write evidence crops for emitted entry events.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory where evidence crops will be written when --write-evidence is enabled.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=DEFAULT_EVIDENCE_CROP_MARGIN,
        help="Extra crop margin as a fraction of box width/height for saved evidence.",
    )
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=DEFAULT_EVIDENCE_PRE_FRAMES,
        help="How many recent crops to keep before an entry event when writing evidence.",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=DEFAULT_EVIDENCE_POST_FRAMES,
        help="How many crops to save after an entry event when writing evidence.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show the live preview window with overlays.",
    )
    return parser


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
    collector = None
    if args.write_evidence:
        collector = EvidenceCollector(
            evidence_dir=args.evidence_dir,
            pre_frames=args.pre_frames,
            post_frames=args.post_frames,
        )

    device = dai.Device()
    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()} Platform: {platform}")

    entrance_states: Dict[int, EntranceState] = {}
    frame_index = 0

    with dai.Pipeline(device) as pipeline:
        print("Unified live pipeline running.")

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
            frame_index += 1
            msg = queue.get()
            frame = msg.getCvFrame()

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            if collector is not None:
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
                if collector is not None:
                    collector.record_entry_event(track_id)

            if args.show_preview:
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
                        states=entrance_states,
                        axis=args.line_axis,
                        line_position=args.line_position,
                        outside_side=args.outside_side,
                    )
                if collector is not None:
                    draw_evidence_status(frame, collector)

                cv2.imshow("Final Pipeline Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Exiting...")
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
