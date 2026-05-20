import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

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
from pipeline.body_evidence import BodyEvidenceExtractor, add_body_evidence_args, build_body_evidence_extractor
from pipeline.depth import (
    CameraIntrinsics,
    DepthEntranceState,
    colorize_depth,
    draw_depth_samples,
    plane_enter_direction_from_args,
    plane_from_args,
    process_depth_entrance_logic,
    process_depth_plane_logic,
    resolve_plane_json_path,
)
from pipeline.detection import PersonDetector, build_person_detector
from pipeline.face_identity import (
    FaceRecognizer,
    add_face_identity_args,
    build_face_recognizer,
    draw_recognized_faces,
)
from pipeline.rgbd_recording import (
    DEFAULT_PLANE_CALIBRATIONS_DIR,
    DEFAULT_RGBD_RECORDINGS_DIR,
    RGBDReplayStream,
    load_rgbd_recording,
    resolve_recording_dir,
)
from pipeline.tracking import SimpleIoUTracker, draw_tracks
from pipeline.visit_identity import (
    add_visit_identity_args,
    draw_visit_labels,
)
from pipeline.visit_registry import (
    VisitRegistry,
    add_visit_registry_args,
    build_track_observations,
)


@dataclass
class SyncedStreamState:
    stream: RGBDReplayStream
    intrinsics: CameraIntrinsics
    tracker: SimpleIoUTracker
    depth_states: dict[int, DepthEntranceState] = field(default_factory=dict)
    plane: object | None = None
    plane_enter_direction: str | None = None
    last_processed_frame_index: int | None = None
    cached_rgb_overlay: np.ndarray | None = None
    camera_role: str = "entrance"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay multiple recorded RGBD streams in sync using recorded RGB timestamps."
    )
    parser.add_argument(
        "--device-id",
        type=str,
        nargs="+",
        required=True,
        help="One or more OAK device ids/MXIDs to replay together.",
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=DEFAULT_RGBD_RECORDINGS_DIR,
        help="Root directory containing RGBD recording folders named oak_<device-id>.rgbd.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 1.0 is real-time based on recorded timestamps.",
    )
    parser.add_argument(
        "--start-mode",
        choices=["overlap", "full"],
        default="overlap",
        help="Start at the overlapping interval or include pre-overlap lead-in with earliest frames.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="How many columns to use when tiling multiple streams in the replay windows.",
    )
    parser.add_argument(
        "--hide-depth-window",
        action="store_true",
        help="Show only the synchronized RGB window.",
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
    parser.add_argument(
        "--plane-json",
        type=Path,
        default=None,
        help=(
            "Optional explicit plane-fit JSON to use for every stream. "
            "When omitted in plane mode, each device auto-loads its own calibration."
        ),
    )
    parser.add_argument(
        "--depth-trigger-mode",
        choices=["threshold", "plane"],
        default="threshold",
        help="Use simple depth thresholding or 3D plane crossing for entry detection.",
    )
    parser.add_argument(
        "--depth-threshold-mm",
        type=int,
        default=2000,
        help="Entry threshold in millimeters for threshold mode.",
    )
    parser.add_argument(
        "--depth-hysteresis-mm",
        type=int,
        default=250,
        help="Rearm hysteresis in millimeters for threshold mode.",
    )
    parser.add_argument(
        "--plane-point-x-mm",
        type=float,
        default=0.0,
        help="Door plane anchor point X in camera coordinates, millimeters.",
    )
    parser.add_argument(
        "--plane-point-y-mm",
        type=float,
        default=0.0,
        help="Door plane anchor point Y in camera coordinates, millimeters.",
    )
    parser.add_argument(
        "--plane-point-z-mm",
        type=float,
        default=2000.0,
        help="Door plane anchor point Z in camera coordinates, millimeters.",
    )
    parser.add_argument(
        "--plane-normal-x",
        type=float,
        default=0.0,
        help="Door plane normal X component in camera coordinates.",
    )
    parser.add_argument(
        "--plane-normal-y",
        type=float,
        default=0.0,
        help="Door plane normal Y component in camera coordinates.",
    )
    parser.add_argument(
        "--plane-normal-z",
        type=float,
        default=1.0,
        help="Door plane normal Z component in camera coordinates.",
    )
    parser.add_argument(
        "--plane-enter-direction",
        choices=["positive_to_negative", "negative_to_positive"],
        default="positive_to_negative",
        help="Which signed-distance transition counts as entering across the plane.",
    )
    parser.add_argument(
        "--plane-hysteresis-mm",
        type=float,
        default=150.0,
        help="Signed-distance hysteresis for plane-crossing rearm logic.",
    )
    parser.add_argument(
        "--depth-min-valid-pixels",
        type=int,
        default=25,
        help="Minimum number of valid depth pixels required inside the sampling ROI.",
    )
    parser.add_argument(
        "--depth-roi-width-fraction",
        type=float,
        default=0.30,
        help="Sampling ROI width as a fraction of tracked box width.",
    )
    parser.add_argument(
        "--depth-roi-height-fraction",
        type=float,
        default=0.22,
        help="Sampling ROI height as a fraction of tracked box height.",
    )
    parser.add_argument(
        "--max-window-width",
        type=int,
        default=1600,
        help="Maximum display width for replay windows after auto-scaling the tiled view.",
    )
    parser.add_argument(
        "--max-window-height",
        type=int,
        default=900,
        help="Maximum display height for replay windows after auto-scaling the tiled view.",
    )
    add_face_identity_args(parser)
    add_body_evidence_args(parser)
    add_visit_identity_args(parser)
    add_visit_registry_args(parser)
    return parser


def render_stream_frame(
    *,
    stream: RGBDReplayStream,
    label: str,
    target_host_seconds: float,
    mode: str,
    source_frame: np.ndarray | None = None,
) -> np.ndarray:
    info = stream.info
    if source_frame is not None:
        source = source_frame
    elif mode == "rgb":
        source = stream.current_rgb_frame
    else:
        source = (
            None
            if stream.current_depth_frame is None
            else colorize_depth(stream.current_depth_frame)
        )

    if source is None or stream.current_frame_meta is None:
        frame = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"{label}: no frame",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    return source.copy()


def fit_row_height(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    target_height = min(frame.shape[0] for frame in frames)
    result: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[0] == target_height:
            result.append(frame)
            continue
        scale = target_height / frame.shape[0]
        target_width = int(round(frame.shape[1] * scale))
        result.append(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA))
    return result


def tile_frames(frames: list[np.ndarray], columns: int) -> np.ndarray:
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    safe_columns = max(1, columns)
    row_count = int(math.ceil(len(frames) / safe_columns))
    rows: list[np.ndarray] = []

    max_width = max(frame.shape[1] for frame in frames)
    max_height = max(frame.shape[0] for frame in frames)
    blank = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    for row_index in range(row_count):
        row_frames = frames[row_index * safe_columns : (row_index + 1) * safe_columns]
        padded_frames = row_frames + [blank.copy() for _ in range(safe_columns - len(row_frames))]
        resized = fit_row_height(padded_frames)
        row = np.hstack(resized)
        rows.append(row)

    target_width = max(row.shape[1] for row in rows)
    normalized_rows: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] == target_width:
            normalized_rows.append(row)
            continue
        pad_width = target_width - row.shape[1]
        normalized_rows.append(
            cv2.copyMakeBorder(
                row,
                top=0,
                bottom=0,
                left=0,
                right=pad_width,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        )
    return np.vstack(normalized_rows)


def fit_to_window(frame: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width and height <= max_height:
        return frame

    width_scale = max_width / width
    height_scale = max_height / height
    scale = min(width_scale, height_scale)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_stream_states(
    *,
    streams: list[RGBDReplayStream],
    args: argparse.Namespace,
    camera_roles: list[str],
) -> list[SyncedStreamState]:
    result: list[SyncedStreamState] = []
    for stream, camera_role in zip(streams, camera_roles):
        info = stream.info
        if info.rgb_intrinsics is None:
            raise RuntimeError(
                f"RGBD recording for device {info.device_id} does not contain RGB intrinsics."
            )

        intrinsics = CameraIntrinsics(
            fx=float(info.rgb_intrinsics["fx"]),
            fy=float(info.rgb_intrinsics["fy"]),
            cx=float(info.rgb_intrinsics["cx"]),
            cy=float(info.rgb_intrinsics["cy"]),
        )

        tracker = SimpleIoUTracker(
            iou_threshold=args.iou_threshold,
            max_missed=args.max_missed,
        )
        state = SyncedStreamState(
            stream=stream,
            intrinsics=intrinsics,
            tracker=tracker,
            camera_role=camera_role,
        )

        if args.depth_trigger_mode == "plane":
            args_for_stream = argparse.Namespace(**vars(args))
            args_for_stream.plane_json = resolve_plane_json_path(
                plane_json=args.plane_json,
                device_id=info.device_id,
                calibrations_root=DEFAULT_PLANE_CALIBRATIONS_DIR,
                recording_dir=info.recording_dir,
            )
            if args_for_stream.plane_json is None:
                raise FileNotFoundError(
                    "Plane mode requested, but no plane JSON was provided and "
                    f"no calibration was found for device {info.device_id}."
                )
            state.plane = plane_from_args(args_for_stream)
            state.plane_enter_direction = plane_enter_direction_from_args(args_for_stream)
            print(f"Loaded plane for {info.device_id} from {args_for_stream.plane_json}")

        result.append(state)
    return result


def resolve_camera_roles(args: argparse.Namespace) -> list[str]:
    if args.camera_role is None or len(args.camera_role) == 0:
        return ["entrance" for _device_id in args.device_id]
    if len(args.camera_role) != len(args.device_id):
        raise ValueError(
            "--camera-role must be omitted or provide exactly one role per --device-id."
        )
    return list(args.camera_role)


def main() -> None:
    args = build_argparser().parse_args()
    camera_roles = resolve_camera_roles(args)
    recordings = [
        load_rgbd_recording(
            resolve_recording_dir(
                recording_dir=None,
                device_id=device_id,
                recordings_root=args.recordings_root,
            )
        )
        for device_id in args.device_id
    ]
    streams = [RGBDReplayStream(info) for info in recordings]
    detector = build_person_detector(args)
    face_matcher = build_face_recognizer(args)
    body_evidence_extractor = build_body_evidence_extractor(args)
    visit_registry = VisitRegistry(
        entrance_merge_window_seconds=args.entrance_merge_window_seconds,
        observer_match_threshold=(
            args.visit_match_threshold
            if args.observer_match_threshold is None
            else args.observer_match_threshold
        ),
        observer_visit_max_age_seconds=args.observer_visit_max_age_seconds,
        log_decisions=args.log_visit_decisions,
    )
    stream_states = build_stream_states(streams=streams, args=args, camera_roles=camera_roles)

    try:
        starts = [info.frames[0].rgb_host_synced_seconds for info in recordings]
        ends = [info.frames[-1].rgb_host_synced_seconds for info in recordings]

        if args.start_mode == "overlap":
            replay_start = max(starts)
            replay_end = min(ends)
        else:
            replay_start = min(starts)
            replay_end = max(ends)

        if replay_end <= replay_start:
            raise RuntimeError("The RGBD recordings do not have an overlapping replay interval.")

        for state in stream_states:
            state.stream.advance_until(replay_start)

        print(f"Replay start rgb_host_synced_seconds={replay_start:.3f}")
        print(f"Replay end rgb_host_synced_seconds={replay_end:.3f}")
        print(
            "Camera roles: "
            + ", ".join(
                f"{recording.device_id}={role}" for recording, role in zip(recordings, camera_roles)
            )
        )
        print("Controls: q=quit, space=pause/resume, ]=faster, [=slower")

        cv2.namedWindow("Synchronized RGBD Replay - RGB", cv2.WINDOW_NORMAL)
        if not args.hide_depth_window:
            cv2.namedWindow("Synchronized RGBD Replay - Depth", cv2.WINDOW_NORMAL)

        paused = False
        speed = args.speed
        started_monotonic = time.monotonic()
        paused_elapsed = 0.0

        while True:
            if paused:
                target_time = replay_start + paused_elapsed * speed
            else:
                elapsed = time.monotonic() - started_monotonic
                paused_elapsed = elapsed
                target_time = replay_start + elapsed * speed

            if target_time > replay_end:
                print("Replay complete.")
                break

            for state in stream_states:
                state.stream.advance_until(target_time)

            rgb_frames = [
                render_stream_frame(
                    stream=state.stream,
                    label=f"Camera {index + 1}",
                    target_host_seconds=target_time,
                    mode="rgb",
                    source_frame=build_processed_rgb_frame(
                        state=state,
                        detector=detector,
                        face_matcher=face_matcher,
                        body_evidence_extractor=body_evidence_extractor,
                        visit_registry=visit_registry,
                        args=args,
                    ),
                )
                for index, state in enumerate(stream_states)
            ]
            rgb_grid = tile_frames(rgb_frames, args.columns)
            rgb_display = fit_to_window(
                rgb_grid,
                max_width=args.max_window_width,
                max_height=args.max_window_height,
            )
            cv2.imshow("Synchronized RGBD Replay - RGB", rgb_display)

            if not args.hide_depth_window:
                depth_frames = [
                    render_stream_frame(
                        stream=state.stream,
                        label=f"Camera {index + 1}",
                        target_host_seconds=target_time,
                        mode="depth",
                    )
                    for index, state in enumerate(stream_states)
                ]
                depth_grid = tile_frames(depth_frames, args.columns)
                depth_display = fit_to_window(
                    depth_grid,
                    max_width=args.max_window_width,
                    max_height=args.max_window_height,
                )
                cv2.imshow("Synchronized RGBD Replay - Depth", depth_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                if not paused:
                    started_monotonic = time.monotonic() - paused_elapsed
            if key == ord("]"):
                speed = min(speed * 1.25, 16.0)
                started_monotonic = time.monotonic() - paused_elapsed
            if key == ord("["):
                speed = max(speed / 1.25, 0.1)
                started_monotonic = time.monotonic() - paused_elapsed
    finally:
        for stream in streams:
            stream.close()
        cv2.destroyAllWindows()


def build_processed_rgb_frame(
    *,
    state: SyncedStreamState,
    detector: PersonDetector,
    face_matcher: FaceRecognizer | None,
    body_evidence_extractor: BodyEvidenceExtractor,
    visit_registry: VisitRegistry,
    args: argparse.Namespace,
) -> np.ndarray:
    rgb_frame = state.stream.current_rgb_frame
    depth_frame_mm = state.stream.current_depth_frame
    if rgb_frame is None:
        return np.zeros((state.stream.info.height, state.stream.info.width, 3), dtype=np.uint8)
    if (
        state.stream.current_frame_meta is not None
        and state.last_processed_frame_index == state.stream.current_frame_meta.frame_index
        and state.cached_rgb_overlay is not None
    ):
        return state.cached_rgb_overlay.copy()

    overlay = rgb_frame.copy()
    if depth_frame_mm is None:
        cv2.putText(
            overlay,
            "No aligned depth for current frame",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    detections = detector.detect(rgb_frame)
    tracks = state.tracker.update(detections)

    if args.depth_trigger_mode == "plane":
        entered_track_ids, depth_samples, signed_distances_mm = process_depth_plane_logic(
            tracks=tracks,
            depth_frame_mm=depth_frame_mm,
            intrinsics=state.intrinsics,
            states=state.depth_states,
            plane=state.plane,
            plane_enter_direction=str(state.plane_enter_direction),
            plane_hysteresis_mm=float(args.plane_hysteresis_mm),
            min_valid_pixels=args.depth_min_valid_pixels,
            roi_width_fraction=args.depth_roi_width_fraction,
            roi_height_fraction=args.depth_roi_height_fraction,
        )
    else:
        entered_track_ids, depth_samples = process_depth_entrance_logic(
            tracks=tracks,
            depth_frame_mm=depth_frame_mm,
            intrinsics=state.intrinsics,
            states=state.depth_states,
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
    host_seconds = (
        0.0
        if state.stream.current_frame_meta is None
        else state.stream.current_frame_meta.rgb_host_synced_seconds
    )
    observations = build_track_observations(
        device_id=state.stream.info.device_id,
        host_seconds=host_seconds,
        frame=rgb_frame,
        tracks=tracks,
        depth_samples=depth_samples,
        recognized_faces=recognized_faces,
        observation_type=state.camera_role,
        body_evidence_by_track=body_evidence_by_track,
    )
    visit_assignments = {}
    if state.camera_role == "observer":
        for track_id, observation in observations.items():
            decision = visit_registry.assign_observer_observation(observation)
            visit_assignments[track_id] = decision.assignment
    else:
        for track_id, observation in observations.items():
            decision = visit_registry.assign_existing_track(observation)
            if decision is not None:
                visit_assignments[track_id] = decision.assignment

    for track_id in entered_track_ids:
        sample = depth_samples.get(track_id)
        if sample is None:
            continue
        observation = observations.get(track_id)
        if state.camera_role == "entrance" and observation is not None:
            decision = visit_registry.assign_entrance_observation(observation)
            visit_assignments[track_id] = decision.assignment
        visit_assignment = visit_assignments.get(track_id)
        if args.depth_trigger_mode == "plane":
            print(
                f"SYNC_DEPTH_PLANE_ENTRY_EVENT device_id={state.stream.info.device_id} "
                f"track_id={track_id} "
                f"visit_id={None if visit_assignment is None else visit_assignment.visit_id} "
                f"host_synced_seconds="
                f"{state.stream.current_frame_meta.rgb_host_synced_seconds:.3f} "
                f"plane_mm={signed_distances_mm.get(track_id, float('nan')):.0f} "
                f"depth_mm={sample.depth_mm:.0f}"
            )
        else:
            print(
                f"SYNC_DEPTH_ENTRY_EVENT device_id={state.stream.info.device_id} "
                f"track_id={track_id} "
                f"visit_id={None if visit_assignment is None else visit_assignment.visit_id} "
                f"host_synced_seconds="
                f"{state.stream.current_frame_meta.rgb_host_synced_seconds:.3f} "
                f"depth_mm={sample.depth_mm:.0f}"
            )

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
    state.last_processed_frame_index = (
        None if state.stream.current_frame_meta is None else state.stream.current_frame_meta.frame_index
    )
    state.cached_rgb_overlay = overlay.copy()
    return overlay


if __name__ == "__main__":
    main()
