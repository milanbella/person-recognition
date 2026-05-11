import argparse
import json
from pathlib import Path
from typing import Dict

import cv2

from pipeline.config import (
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_SCRFD_MODEL,
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
)
from pipeline.depth import (
    CameraIntrinsics,
    DepthEntranceState,
    add_depth_entrance_args,
    colorize_depth,
    draw_depth_event_banner,
    draw_depth_samples,
    plane_enter_direction_from_args,
    plane_from_args,
    process_depth_plane_logic,
    process_depth_entrance_logic,
    resolve_plane_json_path,
)
from pipeline.detection import ScrfdInsightFaceDetector
from pipeline.rgbd_recording import (
    DEFAULT_PLANE_CALIBRATIONS_DIR,
    add_rgbd_recording_lookup_args,
    load_depth_png,
    load_rgbd_recording,
    resolve_recording_dir,
)
from pipeline.tracking import SimpleIoUTracker, draw_tracks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one recorded RGBD stream through depth-based entrance logic."
    )
    add_rgbd_recording_lookup_args(parser)
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help="Optional output path for logged replayed depth entrance events.",
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
        help="Optional override for the host-side SCRFD ONNX model.",
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
    add_depth_entrance_args(parser)
    return parser


def build_event_log_path(recording_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return recording_dir / "depth_events.jsonl"


def main() -> None:
    args = build_argparser().parse_args()
    recording_dir = resolve_recording_dir(
        recording_dir=args.recording_dir,
        device_id=args.device_id,
        recordings_root=args.recordings_root,
    )
    recording = load_rgbd_recording(recording_dir)
    if args.depth_trigger_mode == "plane":
        args.plane_json = resolve_plane_json_path(
            plane_json=args.plane_json,
            device_id=recording.device_id,
            calibrations_root=DEFAULT_PLANE_CALIBRATIONS_DIR,
            recording_dir=recording.recording_dir,
        )
        if args.plane_json is None:
            raise FileNotFoundError(
                "Plane mode requested, but no plane JSON was provided and "
                f"neither {DEFAULT_PLANE_CALIBRATIONS_DIR / ('plane_fit_' + recording.device_id + '.json')} "
                f"nor {recording.recording_dir / 'plane_fit.json'} was found."
            )
    if recording.rgb_intrinsics is None:
        raise RuntimeError(
            "This RGBD recording does not contain RGB intrinsics. Re-record with the current record_rgbd_stream.py."
        )
    rgb_intrinsics = CameraIntrinsics(
        fx=float(recording.rgb_intrinsics["fx"]),
        fy=float(recording.rgb_intrinsics["fy"]),
        cx=float(recording.rgb_intrinsics["cx"]),
        cy=float(recording.rgb_intrinsics["cy"]),
    )
    plane = plane_from_args(args)
    plane_enter_direction = plane_enter_direction_from_args(args)

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
    depth_states: Dict[int, DepthEntranceState] = {}

    capture = cv2.VideoCapture(str(recording.rgb_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open RGB video: {recording.rgb_video_path}")

    event_log_path = build_event_log_path(recording.recording_dir, args.event_log)
    event_log = event_log_path.open("w", encoding="utf-8")
    event_log.write(
        json.dumps(
            {
                "type": "replay_header",
                "recording_dir": str(recording.recording_dir.resolve()),
                "device_id": recording.device_id,
                "depth_threshold_mm": args.depth_threshold_mm,
                "depth_hysteresis_mm": args.depth_hysteresis_mm,
                "depth_min_valid_pixels": args.depth_min_valid_pixels,
                "depth_roi_width_fraction": args.depth_roi_width_fraction,
                "depth_roi_height_fraction": args.depth_roi_height_fraction,
                "score_threshold": args.score_threshold,
                "iou_threshold": args.iou_threshold,
                "max_missed": args.max_missed,
                "depth_trigger_mode": args.depth_trigger_mode,
                "speed": args.speed,
                "plane_json": None if args.plane_json is None else str(args.plane_json.resolve()),
                "plane_enter_direction": plane_enter_direction,
            }
        )
        + "\n"
    )

    print(f"Replaying depth recording from {recording.recording_dir}")
    print(f"Writing event log to {event_log_path}")
    if args.depth_trigger_mode == "plane" and args.plane_json is not None:
        print(f"Loaded plane from {args.plane_json}")
    print("Controls: q=quit, space=pause/resume, ]=faster, [=slower")

    paused = False
    speed = args.speed
    event_count = 0
    frame_idx = 0
    event_flash_remaining = 0
    event_flash_text = ""

    try:
        while frame_idx < len(recording.frames):
            frame_meta = recording.frames[frame_idx]
            ok, rgb_frame = capture.read()
            if not ok or rgb_frame is None:
                print("RGB replay ended.")
                break

            depth_frame_mm = load_depth_png(recording, frame_meta)
            detections = detector.detect(rgb_frame)
            tracks = tracker.update(detections)

            if args.depth_trigger_mode == "plane":
                entered_track_ids, depth_samples, signed_distances_mm = process_depth_plane_logic(
                    tracks=tracks,
                    depth_frame_mm=depth_frame_mm,
                    intrinsics=rgb_intrinsics,
                    states=depth_states,
                    plane=plane,
                    plane_enter_direction=plane_enter_direction,
                    plane_hysteresis_mm=float(args.plane_hysteresis_mm),
                    min_valid_pixels=args.depth_min_valid_pixels,
                    roi_width_fraction=args.depth_roi_width_fraction,
                    roi_height_fraction=args.depth_roi_height_fraction,
                )
            else:
                entered_track_ids, depth_samples = process_depth_entrance_logic(
                    tracks=tracks,
                    depth_frame_mm=depth_frame_mm,
                    intrinsics=rgb_intrinsics,
                    states=depth_states,
                    depth_threshold_mm=float(args.depth_threshold_mm),
                    depth_hysteresis_mm=float(args.depth_hysteresis_mm),
                    min_valid_pixels=args.depth_min_valid_pixels,
                    roi_width_fraction=args.depth_roi_width_fraction,
                    roi_height_fraction=args.depth_roi_height_fraction,
                )
                signed_distances_mm = {}

            for track_id in entered_track_ids:
                sample = depth_samples.get(track_id)
                if sample is None:
                    continue
                event_count += 1
                event_payload = {
                    "type": "depth_plane_entry_event"
                    if args.depth_trigger_mode == "plane"
                    else "depth_entry_event",
                    "device_id": recording.device_id,
                    "track_id": track_id,
                    "frame_index": frame_meta.frame_index,
                    "rgb_sequence_num": frame_meta.rgb_sequence_num,
                    "rgb_host_synced_seconds": frame_meta.rgb_host_synced_seconds,
                    "depth_sequence_num": frame_meta.depth_sequence_num,
                    "depth_host_synced_seconds": frame_meta.depth_host_synced_seconds,
                    "matched_depth_delta_ms": frame_meta.matched_depth_delta_ms,
                    "depth_mm": sample.depth_mm,
                }
                if args.depth_trigger_mode == "plane":
                    event_payload["plane_signed_distance_mm"] = signed_distances_mm.get(track_id)
                event_log.write(json.dumps(event_payload) + "\n")
                event_log.flush()
                if args.depth_trigger_mode == "plane":
                    print(
                        f"DEPTH_PLANE_ENTRY_EVENT track_id={track_id} "
                        f"host_synced_seconds={frame_meta.rgb_host_synced_seconds:.3f} "
                        f"plane_mm={signed_distances_mm.get(track_id, float('nan')):.0f} "
                        f"depth_mm={sample.depth_mm:.0f}"
                    )
                else:
                    print(
                        f"DEPTH_ENTRY_EVENT track_id={track_id} "
                        f"host_synced_seconds={frame_meta.rgb_host_synced_seconds:.3f} "
                        f"depth_mm={sample.depth_mm:.0f}"
                    )
            if entered_track_ids:
                prefix = "PLANE ENTRY" if args.depth_trigger_mode == "plane" else "DEPTH ENTRY"
                event_flash_text = f"{prefix}: " + ", ".join(
                    str(track_id) for track_id in entered_track_ids
                )
                event_flash_remaining = 12

            overlay = rgb_frame.copy()
            draw_tracks(overlay, tracks)
            draw_depth_samples(
                overlay,
                tracks=tracks,
                depth_samples=depth_samples,
                depth_threshold_mm=float(args.depth_threshold_mm),
                signed_distances_mm=signed_distances_mm,
                plane_mode=args.depth_trigger_mode == "plane",
            )
            if event_flash_remaining > 0:
                draw_depth_event_banner(overlay, event_flash_text)
                event_flash_remaining -= 1
            cv2.putText(
                overlay,
                (
                    f"frame={frame_meta.frame_index} host_synced="
                    f"{frame_meta.rgb_host_synced_seconds:.3f}s speed={speed:.2f}x"
                ),
                (20, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Depth Replay Tuner", overlay)
            if args.show_depth_window:
                cv2.imshow("Depth Replay Aligned Depth", colorize_depth(depth_frame_mm))

            if frame_idx + 1 < len(recording.frames):
                next_meta = recording.frames[frame_idx + 1]
                delta_seconds = max(
                    0.0,
                    next_meta.rgb_host_synced_seconds - frame_meta.rgb_host_synced_seconds,
                )
                wait_ms = max(1, int(round((delta_seconds / speed) * 1000.0)))
            else:
                wait_ms = 1

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
                frame_idx += 1
            else:
                continue
    finally:
        event_log.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Logged {event_count} depth events to {event_log_path}")


if __name__ == "__main__":
    main()
