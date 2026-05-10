import argparse
import json
import time
from pathlib import Path
from typing import Dict

import cv2

from pipeline.config import (
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_ENTRANCE_LINE_AXIS,
    DEFAULT_ENTRANCE_LINE_POSITION,
    DEFAULT_ENTRANCE_MIN_HISTORY,
    DEFAULT_OUTSIDE_SIDE,
    DEFAULT_SCRFD_MODEL,
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
)
from pipeline.detection import ScrfdInsightFaceDetector
from pipeline.entrance import (
    EntranceState,
    draw_entry_events,
    draw_entrance_debug,
    draw_entrance_line,
    process_entrance_logic,
)
from pipeline.recording import ReplayStream, load_recording, resolve_timestamps_path
from pipeline.tracking import SimpleIoUTracker, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one recorded stream through detection, tracking, and entrance-event logic."
    )
    parser.add_argument("--video", type=Path, required=True, help="Recorded .avi file.")
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="Optional .timestamps.jsonl path. Defaults to the video sidecar path.",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help="Optional output path for logged replayed entrance events.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 1.0 uses recorded frame timing.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_SCRFD_MODEL,
        help="Path to the host-side SCRFD ONNX model.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=DEFAULT_DETECTION_INPUT_WIDTH,
        help="Detector input width.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=DEFAULT_DETECTION_INPUT_HEIGHT,
        help="Detector input height.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_DETECTION_SCORE_THRESHOLD,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=DEFAULT_DETECTION_NMS_THRESHOLD,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_TRACKING_IOU_THRESHOLD,
        help="Minimum IoU for matching a detection to an existing track.",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=DEFAULT_TRACKING_MAX_MISSED,
        help="How many consecutive frames a track may be unmatched before removal.",
    )
    parser.add_argument(
        "--line-axis",
        choices=["x", "y"],
        default=DEFAULT_ENTRANCE_LINE_AXIS,
        help="Axis for the entrance line.",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=DEFAULT_ENTRANCE_LINE_POSITION,
        help="Normalized line position in the frame, between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--outside-side",
        choices=["less", "greater"],
        default=DEFAULT_OUTSIDE_SIDE,
        help="Which side of the line is considered outside.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_ENTRANCE_MIN_HISTORY,
        help="Minimum centroid history length before an entry event may be emitted.",
    )
    parser.add_argument(
        "--debug-entrance",
        action="store_true",
        help="Show centroid/side debug overlays and print side transitions.",
    )
    return parser


def build_event_log_path(video_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return video_path.with_suffix(".entrance_events.jsonl")


def main() -> None:
    args = build_argparser().parse_args()
    timestamps_path = resolve_timestamps_path(args.video, args.timestamps)
    recording = load_recording(args.video, timestamps_path)
    replay = ReplayStream(recording)
    event_log_path = build_event_log_path(args.video, args.event_log)

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
    entrance_states: Dict[int, EntranceState] = {}

    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    event_log = event_log_path.open("w", encoding="utf-8")
    event_log.write(
        json.dumps(
            {
                "type": "replay_header",
                "video_path": str(args.video.resolve()),
                "timestamps_path": str(timestamps_path.resolve()),
                "device_id": recording.device_id,
                "line_axis": args.line_axis,
                "line_position": args.line_position,
                "outside_side": args.outside_side,
                "min_history": args.min_history,
                "speed": args.speed,
            }
        )
        + "\n"
    )

    print(f"Replaying {args.video.name} for entrance tuning.")
    print(f"Writing event log to {event_log_path}")
    print("Controls: q=quit, space=pause/resume, ]=faster, [=slower")

    paused = False
    speed = args.speed
    event_count = 0

    try:
        while replay.current_frame is not None and replay.current_stamp is not None:
            frame = replay.current_frame.copy()
            stamp = replay.current_stamp

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
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
                event_count += 1
                payload = {
                    "type": "entry_event",
                    "device_id": recording.device_id,
                    "track_id": track_id,
                    "frame_index": stamp.frame_index,
                    "sequence_num": stamp.sequence_num,
                    "host_synced_seconds": stamp.host_synced_seconds,
                    "device_monotonic_seconds": stamp.device_monotonic_seconds,
                    "received_utc": stamp.received_utc,
                    "line_axis": args.line_axis,
                    "line_position": args.line_position,
                    "outside_side": args.outside_side,
                }
                event_log.write(json.dumps(payload) + "\n")
                event_log.flush()
                print(
                    f"ENTRY_EVENT track_id={track_id} frame_index={stamp.frame_index} "
                    f"host_synced_seconds={stamp.host_synced_seconds:.3f}"
                )

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

            cv2.putText(
                frame,
                f"device={recording.device_id} frame={stamp.frame_index} speed={speed:.2f}x",
                (20, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"host_synced={stamp.host_synced_seconds:.3f}s line={args.line_axis}:{args.line_position:.3f}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Replay Entrance Tuner", frame)

            next_stamp = (
                recording.frames[replay.next_index]
                if replay.next_index < len(recording.frames)
                else None
            )
            wait_ms = 1
            if next_stamp is not None:
                delta_seconds = max(
                    0.0,
                    next_stamp.host_synced_seconds - stamp.host_synced_seconds,
                )
                wait_ms = max(1, int(round((delta_seconds / speed) * 1000.0)))

            key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("]"):
                speed = min(speed * 1.25, 16.0)
            if key == ord("["):
                speed = max(speed / 1.25, 0.1)

            if not paused:
                if not replay.advance():
                    print("Replay complete.")
                    break
            else:
                continue
    finally:
        event_log.close()
        replay.close()
        cv2.destroyAllWindows()

    print(f"Logged {event_count} entry events to {event_log_path}")


if __name__ == "__main__":
    main()
