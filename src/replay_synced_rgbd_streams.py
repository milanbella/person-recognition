import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.config import (
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_PERSON_DETECTOR_BACKEND,
    DEFAULT_PERSON_DETECTOR_MODEL,
    DEFAULT_PERSON_TRACKER_BACKEND,
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
from pipeline.tracking import PersonTracker, build_person_tracker, draw_tracks
from pipeline.visit_identity import (
    add_visit_identity_args,
    draw_visit_labels,
)
from pipeline.visit_registry import (
    CAMERA_ROLE_ENTRANCE,
    FrameEvidence,
    ShopVisit,
    TrackVisitEvidence,
    VisitRegistry,
    VisitRegistryDecision,
    add_visit_registry_args,
    build_track_visit_evidence,
    is_entrance_enabled,
    is_observer_enabled,
)


@dataclass
class SyncedStreamState:
    stream: RGBDReplayStream
    intrinsics: CameraIntrinsics
    tracker: PersonTracker
    depth_states: dict[int, DepthEntranceState] = field(default_factory=dict)
    plane: object | None = None
    plane_enter_direction: str | None = None
    last_processed_frame_index: int | None = None
    cached_rgb_overlay: np.ndarray | None = None
    camera_role: str = "entrance"


class ReplayArtifactWriter:
    def __init__(self, output_dir: Path | None) -> None:
        self.output_dir = output_dir
        self.track_evidence_file = None
        self.visit_decisions_file = None
        self.entrance_events_file = None
        if output_dir is None:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        self.track_evidence_file = (output_dir / "track_visit_evidence.jsonl").open("w", encoding="utf-8")
        self.visit_decisions_file = (output_dir / "visit_decisions.jsonl").open("w", encoding="utf-8")
        self.entrance_events_file = (output_dir / "entrance_events.jsonl").open("w", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def write_replay_config(
        self,
        *,
        args: argparse.Namespace,
        recordings: list[Any],
        camera_roles: list[str],
    ) -> None:
        if self.output_dir is None:
            return
        payload = {
            "device_ids": list(args.device_id),
            "camera_roles": camera_roles,
            "recordings": [
                {
                    "device_id": recording.device_id,
                    "recording_dir": str(recording.recording_dir),
                    "frame_count": len(recording.frames),
                    "start_rgb_host_synced_seconds": recording.frames[0].rgb_host_synced_seconds,
                    "end_rgb_host_synced_seconds": recording.frames[-1].rgb_host_synced_seconds,
                }
                for recording in recordings
            ],
            "args": _json_safe_namespace(args),
        }
        (self.output_dir / "replay_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_track_evidence(self, track_evidence: TrackVisitEvidence) -> None:
        if self.track_evidence_file is None:
            return
        self.track_evidence_file.write(json.dumps(_track_evidence_to_dict(track_evidence)) + "\n")

    def write_visit_decision(
        self,
        *,
        resolution: str,
        track_evidence: TrackVisitEvidence,
        decision: VisitRegistryDecision,
    ) -> None:
        if self.visit_decisions_file is None:
            return
        payload = {
            "resolution": resolution,
            "track_evidence": _track_evidence_to_dict(track_evidence),
            "decision": _decision_to_dict(decision),
        }
        self.visit_decisions_file.write(json.dumps(payload) + "\n")

    def write_entrance_event(self, payload: dict[str, Any]) -> None:
        if self.entrance_events_file is None:
            return
        self.entrance_events_file.write(json.dumps(payload) + "\n")

    def write_final_visits(self, visit_registry: VisitRegistry) -> None:
        if self.output_dir is None:
            return
        visits = [_shop_visit_to_dict(visit) for visit in sorted(visit_registry.visits.values(), key=lambda item: item.visit_id)]
        (self.output_dir / "final_visits.json").write_text(
            json.dumps({"visits": visits}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        with (self.output_dir / "final_visits.csv").open("w", encoding="utf-8", newline="") as csv_file:
            fieldnames = [
                "visit_id",
                "origin",
                "created_host_seconds",
                "last_seen_host_seconds",
                "last_device_id",
                "last_track_id",
                "observation_count",
                "observer_observation_count",
                "depth_mm",
                "face_identity_ids",
                "entrance_observation_times",
                "merged_visit_ids",
                "has_body_appearance",
                "body_aspect_ratio",
                "body_height_px",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for visit in visits:
                writer.writerow(_shop_visit_csv_row(visit))

    def close(self) -> None:
        for handle in [self.track_evidence_file, self.visit_decisions_file, self.entrance_events_file]:
            if handle is not None:
                handle.close()


def _json_safe_namespace(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def _track_evidence_to_dict(track_evidence: TrackVisitEvidence) -> dict[str, Any]:
    body_appearance = track_evidence.body_appearance
    return {
        "camera_role": track_evidence.camera_role,
        "device_id": track_evidence.device_id,
        "track_id": track_evidence.track_id,
        "host_seconds": track_evidence.host_seconds,
        "track_bbox": list(track_evidence.track_bbox),
        "face_identity_ids": list(track_evidence.face_identity_ids),
        "depth_mm": track_evidence.depth_mm,
        "has_body_appearance": body_appearance is not None,
        "body_aspect_ratio": None if body_appearance is None else body_appearance.aspect_ratio,
        "body_height_px": None if body_appearance is None else body_appearance.height_px,
    }


def _decision_to_dict(decision: VisitRegistryDecision) -> dict[str, Any]:
    assignment = decision.assignment
    return {
        "visit_id": assignment.visit_id,
        "track_id": assignment.track_id,
        "device_id": assignment.device_id,
        "face_identity_ids": list(assignment.face_identity_ids),
        "matched_score": assignment.matched_score,
        "origin": assignment.origin,
        "decision": decision.decision,
        "reason": decision.reason,
        "score": decision.score,
        "matched_visit_id": decision.matched_visit_id,
        "score_breakdown": decision.score_breakdown,
    }


def _shop_visit_to_dict(visit: ShopVisit) -> dict[str, Any]:
    return {
        "visit_id": visit.visit_id,
        "origin": visit.origin,
        "created_host_seconds": visit.created_host_seconds,
        "last_seen_host_seconds": visit.last_seen_host_seconds,
        "last_device_id": visit.last_device_id,
        "last_track_id": visit.last_track_id,
        "observation_count": visit.observation_count,
        "observer_observation_count": visit.observer_observation_count,
        "depth_mm": visit.depth_mm,
        "face_identity_ids": sorted(visit.face_identity_ids),
        "entrance_observation_times": list(visit.entrance_observation_times),
        "merged_visit_ids": sorted(visit.merged_visit_ids),
        "has_body_appearance": visit.appearance is not None,
        "body_aspect_ratio": None if visit.appearance is None else visit.appearance.aspect_ratio,
        "body_height_px": None if visit.appearance is None else visit.appearance.height_px,
    }


def _shop_visit_csv_row(visit: dict[str, Any]) -> dict[str, Any]:
    row = dict(visit)
    row["face_identity_ids"] = ",".join(str(value) for value in visit.get("face_identity_ids", []))
    row["entrance_observation_times"] = ",".join(
        f"{float(value):.3f}" for value in visit.get("entrance_observation_times", [])
    )
    row["merged_visit_ids"] = ",".join(str(value) for value in visit.get("merged_visit_ids", []))
    return row


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
        "--tracker-backend",
        choices=["iou"],
        default=DEFAULT_PERSON_TRACKER_BACKEND,
        help="Person tracker backend. Current default is simple IoU tracking.",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for synced replay artifacts: decisions, evidence, entrance events, and final visits.",
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

        tracker = build_person_tracker(args)
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
        return [CAMERA_ROLE_ENTRANCE for _device_id in args.device_id]
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
    artifact_writer = ReplayArtifactWriter(args.output_dir)
    artifact_writer.write_replay_config(args=args, recordings=recordings, camera_roles=camera_roles)
    if artifact_writer.enabled:
        print(f"Writing synced replay artifacts to {artifact_writer.output_dir}")

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
                        artifact_writer=artifact_writer,
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
        artifact_writer.write_final_visits(visit_registry)
        artifact_writer.close()
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
    artifact_writer: ReplayArtifactWriter,
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
        depth_result = process_depth_plane_logic(
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
        depth_result = process_depth_entrance_logic(
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
    entered_track_ids = depth_result.entered_track_ids
    depth_samples = depth_result.depth_samples
    signed_distances_mm = depth_result.signed_distances_mm

    recognized_faces = []
    if face_matcher is not None:
        recognized_faces = face_matcher.recognize(rgb_frame, tracks=tracks)
    body_evidence_by_track = body_evidence_extractor.extract(rgb_frame, tracks=tracks)
    host_seconds = (
        0.0
        if state.stream.current_frame_meta is None
        else state.stream.current_frame_meta.rgb_host_synced_seconds
    )
    frame_evidence = FrameEvidence(
        device_id=state.stream.info.device_id,
        host_seconds=host_seconds,
        camera_role=state.camera_role,
        tracks=tracks,
        depth_samples_by_track=depth_samples,
        recognized_faces=recognized_faces,
        body_evidence_by_track=body_evidence_by_track,
    )
    track_visit_evidence_by_id = build_track_visit_evidence(frame_evidence)
    for track_evidence in track_visit_evidence_by_id.values():
        artifact_writer.write_track_evidence(track_evidence)
    visit_assignments = {}
    observer_enabled = is_observer_enabled(state.camera_role)
    entrance_enabled = is_entrance_enabled(state.camera_role)
    for track_id, track_evidence in track_visit_evidence_by_id.items():
        decision = visit_registry.resolve_existing_track(track_evidence)
        resolution = "existing_track"
        if decision is None and observer_enabled:
            decision = visit_registry.resolve_observer_track(track_evidence)
            resolution = "observer_track"
        if decision is not None:
            visit_assignments[track_id] = decision.assignment
            artifact_writer.write_visit_decision(
                resolution=resolution,
                track_evidence=track_evidence,
                decision=decision,
            )

    for track_id in entered_track_ids:
        sample = depth_samples.get(track_id)
        if sample is None:
            continue
        track_evidence = track_visit_evidence_by_id.get(track_id)
        if entrance_enabled and track_evidence is not None:
            decision = visit_registry.resolve_entrance_track(track_evidence)
            visit_assignments[track_id] = decision.assignment
            artifact_writer.write_visit_decision(
                resolution="entrance_track",
                track_evidence=track_evidence,
                decision=decision,
            )
        visit_assignment = visit_assignments.get(track_id)
        event_payload = {
            "type": "sync_depth_plane_entry_event"
            if args.depth_trigger_mode == "plane"
            else "sync_depth_entry_event",
            "device_id": state.stream.info.device_id,
            "camera_role": state.camera_role,
            "track_id": track_id,
            "visit_id": None if visit_assignment is None else visit_assignment.visit_id,
            "host_synced_seconds": None
            if state.stream.current_frame_meta is None
            else state.stream.current_frame_meta.rgb_host_synced_seconds,
            "depth_mm": sample.depth_mm,
            "plane_signed_distance_mm": signed_distances_mm.get(track_id)
            if args.depth_trigger_mode == "plane"
            else None,
        }
        artifact_writer.write_entrance_event(event_payload)
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
