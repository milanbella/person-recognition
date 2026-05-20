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
    DEFAULT_PERSON_DETECTOR_BACKEND,
    DEFAULT_PERSON_DETECTOR_MODEL,
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
)
from pipeline.body_evidence import add_body_evidence_args, build_body_evidence_extractor
from pipeline.depth import (
    CameraIntrinsics,
    DepthEntranceState,
    add_depth_entrance_args,
    colorize_depth,
    draw_depth_samples,
    plane_enter_direction_from_args,
    plane_from_args,
    process_depth_plane_logic,
    process_depth_entrance_logic,
    resolve_plane_json_path,
)
from pipeline.detection import build_person_detector
from pipeline.face_identity import (
    add_face_identity_args,
    build_face_recognizer,
    draw_recognized_faces,
)
from pipeline.rgbd_recording import (
    DEFAULT_PLANE_CALIBRATIONS_DIR,
    add_rgbd_recording_lookup_args,
    load_depth_png,
    load_rgbd_recording,
    resolve_recording_dir,
)
from pipeline.tracking import SimpleIoUTracker, draw_tracks
from pipeline.visit_identity import (
    VisitIdentityManager,
    add_visit_identity_args,
    draw_visit_labels,
)


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
        "--detector-backend",
        choices=["scrfd"],
        default=DEFAULT_PERSON_DETECTOR_BACKEND,
        help="Person detector backend. Current default is SCRFD via InsightFace model zoo.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_PERSON_DETECTOR_MODEL,
        help="Optional override for the host-side person detector ONNX model.",
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
    parser.add_argument(
        "--hide-depth-window",
        action="store_true",
        help="Compatibility option. Depth window is hidden unless --show-depth-window is set.",
    )
    add_face_identity_args(parser)
    add_body_evidence_args(parser)
    add_visit_identity_args(parser)
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

    detector = build_person_detector(args)
    tracker = SimpleIoUTracker(
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
    )
    face_matcher = build_face_recognizer(args)
    body_evidence_extractor = build_body_evidence_extractor(args)
    visit_manager = VisitIdentityManager(
        match_threshold=args.visit_match_threshold,
        same_camera_max_age_seconds=args.visit_same_camera_max_age_seconds,
        cross_camera_max_age_seconds=args.visit_cross_camera_max_age_seconds,
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

            recognized_faces = []
            if face_matcher is not None:
                recognized_faces = face_matcher.recognize(rgb_frame, tracks=tracks)
            body_evidence_by_track = body_evidence_extractor.extract(rgb_frame, tracks=tracks)
            body_appearances = {
                track_id: evidence.appearance
                for track_id, evidence in body_evidence_by_track.items()
                if evidence.appearance is not None
            }
            visit_assignments = visit_manager.update(
                device_id=recording.device_id,
                host_seconds=frame_meta.rgb_host_synced_seconds,
                frame=rgb_frame,
                tracks=tracks,
                depth_samples=depth_samples,
                recognized_faces=recognized_faces,
                body_appearances=body_appearances,
            )

            for track_id in entered_track_ids:
                sample = depth_samples.get(track_id)
                if sample is None:
                    continue
                visit_assignment = visit_assignments.get(track_id)
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
                    "visit_id": None if visit_assignment is None else visit_assignment.visit_id,
                    "face_identity_ids": []
                    if visit_assignment is None
                    else list(visit_assignment.face_identity_ids),
                }
                if args.depth_trigger_mode == "plane":
                    event_payload["plane_signed_distance_mm"] = signed_distances_mm.get(track_id)
                event_log.write(json.dumps(event_payload) + "\n")
                event_log.flush()
                if args.depth_trigger_mode == "plane":
                    print(
                        f"DEPTH_PLANE_ENTRY_EVENT track_id={track_id} "
                        f"visit_id={None if visit_assignment is None else visit_assignment.visit_id} "
                        f"host_synced_seconds={frame_meta.rgb_host_synced_seconds:.3f} "
                        f"plane_mm={signed_distances_mm.get(track_id, float('nan')):.0f} "
                        f"depth_mm={sample.depth_mm:.0f}"
                    )
                else:
                    print(
                        f"DEPTH_ENTRY_EVENT track_id={track_id} "
                        f"visit_id={None if visit_assignment is None else visit_assignment.visit_id} "
                        f"host_synced_seconds={frame_meta.rgb_host_synced_seconds:.3f} "
                        f"depth_mm={sample.depth_mm:.0f}"
                    )
            overlay = rgb_frame.copy()
            draw_tracks(overlay, tracks)
            draw_visit_labels(overlay, tracks, visit_assignments)
            draw_depth_samples(
                overlay,
                tracks=tracks,
                depth_samples=depth_samples,
                depth_threshold_mm=float(args.depth_threshold_mm),
                signed_distances_mm=signed_distances_mm,
                plane_mode=args.depth_trigger_mode == "plane",
            )
            if face_matcher is not None:
                draw_recognized_faces(overlay, recognized_faces)
            cv2.imshow("Depth Replay Tuner", overlay)
            if args.show_depth_window and not args.hide_depth_window:
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
