import argparse
import json
import math
import signal
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np

from pipeline.body_evidence import BodyEvidenceExtractor, add_body_evidence_args, build_body_evidence_extractor
from pipeline.camera import configure_live_device, device_identifier, list_available_devices, print_available_devices
from pipeline.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_PERSON_DETECTOR_BACKEND,
    DEFAULT_PERSON_DETECTOR_MODEL,
    DEFAULT_PERSON_TRACKER_BACKEND,
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)
from pipeline.depth import (
    CameraIntrinsics,
    DepthEntranceState,
    colorize_depth,
    draw_depth_samples,
    intrinsics_from_matrix,
    plane_enter_direction_from_args,
    plane_from_args,
    process_depth_entrance_logic,
    process_depth_plane_logic,
    resolve_plane_json_path,
    add_plane_track_split_recovery_args,
)
from pipeline.detection import DETECTOR_BACKEND_CHOICES, PersonDetector, build_person_detector
from pipeline.face_identity import (
    FaceRecognizer,
    add_face_identity_args,
    build_face_recognizer,
    draw_recognized_faces,
    face_recognition_eligible_tracks,
)
from pipeline.mjpeg_stream_server import MjpegStreamServer
from pipeline.observer_api import ObserverCameraSnapshot, build_observer_camera_snapshot
from pipeline.rgbd_recording import DEFAULT_PLANE_CALIBRATIONS_DIR
from pipeline.shop_state_store import DEFAULT_SHOP_STATE_DB, ShopStateStore
from pipeline.tracking import PersonTracker, build_person_tracker, draw_tracks
from pipeline.visit_identity import add_visit_identity_args, draw_visit_labels
from pipeline.visit_registry import (
    CAMERA_ROLE_ENTRANCE,
    FrameEvidence,
    TrackVisitEvidence,
    VisitRegistry,
    VisitRegistryDecision,
    add_visit_registry_args,
    build_track_visit_evidence,
    is_entrance_enabled,
    is_observer_enabled,
)
from replay_synced_rgbd_streams import (
    ReplayArtifactWriter,
    VisitPlaneState,
    _visit_plane_inside,
    _visit_plane_outside,
    fit_to_window,
    log_plane_trace,
    tile_frames,
)


@dataclass
class LiveDepthPacket:
    sequence_num: int
    host_synced_seconds: float
    device_monotonic_seconds: float
    frame_mm: np.ndarray


@dataclass(frozen=True)
class ProcessedLiveFrame:
    overlay: np.ndarray
    observer_snapshot: ObserverCameraSnapshot | None


@dataclass
class LiveSyncedStreamState:
    device_id: str
    camera_role: str
    device: dai.Device
    pipeline: dai.Pipeline
    rgb_queue: dai.MessageQueue
    depth_queue: dai.MessageQueue
    intrinsics: CameraIntrinsics
    tracker: PersonTracker
    depth_states: dict[int, DepthEntranceState] = field(default_factory=dict)
    visit_plane_states: dict[int, VisitPlaneState] = field(default_factory=dict)
    recent_depth_packets: deque[LiveDepthPacket] = field(default_factory=lambda: deque(maxlen=32))
    plane: object | None = None
    plane_enter_direction: str | None = None
    latest_depth_visual: np.ndarray | None = None
    cached_rgb_raw: np.ndarray | None = None
    cached_rgb_overlay: np.ndarray | None = None
    cached_observer_snapshot: ObserverCameraSnapshot | None = None
    cached_depth_overlay: np.ndarray | None = None
    last_processed_rgb_sequence: int | None = None
    last_rgb_host_synced_seconds: float | None = None


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class ShopApiClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        shop_id: int | None,
        max_age_seconds: int,
        timeout_seconds: float,
    ) -> None:
        self.enabled = bool(base_url)
        self.base_url = "" if base_url is None else base_url.rstrip("/")
        self.api_key = api_key
        self.shop_id = shop_id
        self.max_age_seconds = max_age_seconds
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(NoRedirectHandler)
        self.bound_visit_ids: set[int] = set()
        self.left_visit_ids: set[int] = set()

        if self.enabled and not self.api_key:
            raise ValueError("--shop-api-key is required when --shop-api-base-url is set.")
        if self.enabled and self.shop_id is None:
            raise ValueError("--shop-id is required when --shop-api-base-url is set.")

    def bind_visit(self, visit_id: int | None) -> str | None:
        if not self.enabled or visit_id is None or visit_id in self.bound_visit_ids:
            return None
        latest = self._post(
            "/shop-api/shopping-customer/latest-without-visit-id",
            {
                "shopId": self.shop_id,
                "maxAgeSeconds": self.max_age_seconds,
            },
            not_found_ok=True,
        )
        if latest is None:
            print(
                f"SHOP_API_BIND_SKIPPED visit_id={visit_id} "
                f"reason=no_recent_unbound_customer max_age_seconds={self.max_age_seconds}"
            )
            return None

        customer_id = latest.get("customerId")
        if not customer_id:
            print(f"SHOP_API_BIND_FAILED visit_id={visit_id} reason=missing_customer_id")
            return None

        self._post(
            "/shop-api/shopping-customer/set-visit-id",
            {
                "shopId": self.shop_id,
                "customerId": customer_id,
                "visitId": visit_id,
            },
        )
        self.bound_visit_ids.add(visit_id)
        print(f"SHOP_API_VISIT_BOUND visit_id={visit_id} customer_id={customer_id}")
        return str(customer_id)

    def mark_left(self, visit_id: int | None) -> bool:
        if not self.enabled or visit_id is None or visit_id in self.left_visit_ids:
            return False
        self._post(
            "/shop-api/shopping-customer/mark-left",
            {
                "shopId": self.shop_id,
                "visitId": visit_id,
            },
        )
        self.left_visit_ids.add(visit_id)
        print(f"SHOP_API_VISIT_LEFT visit_id={visit_id}")
        return True

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        not_found_ok: bool = False,
    ) -> dict[str, Any] | None:
        body = json.dumps(payload).encode("utf-8")
        url = self.base_url + path
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": str(self.api_key),
            },
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                location = exc.headers.get("Location", "")
                raise RuntimeError(
                    f"Shop API POST {url} was redirected to {location}. "
                    "Use the final non-redirecting --shop-api-base-url, otherwise Python may replay POST as GET."
                ) from exc
            if not_found_ok and exc.code == 404:
                return None
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Shop API POST {url} failed: {exc.code} {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Shop API POST {url} failed: {exc}") from exc

        if not response_body:
            return {}
        return json.loads(response_body)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live synchronized RGBD processing for multiple OAK cameras."
    )
    parser.add_argument(
        "--device-id",
        type=str,
        nargs="+",
        default=None,
        help="One or more OAK device ids/MXIDs to process together.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available OAK devices and exit.")
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS, help="Camera output FPS.")
    parser.add_argument(
        "--frame-width",
        type=int,
        default=PREVIEW_WIDTH,
        help="RGB camera and raw MJPEG stream width. Default: 3840.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=PREVIEW_HEIGHT,
        help="RGB camera and raw MJPEG stream height. Default: 2160.",
    )
    parser.add_argument("--columns", type=int, default=2, help="Columns for tiled live view.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV windows. Use this for systemd/background service mode.",
    )
    parser.add_argument("--stream-host", default="0.0.0.0", help="MJPEG streaming API bind address.")
    parser.add_argument("--stream-port", type=int, default=8002, help="MJPEG streaming API port.")
    parser.add_argument("--stream-jpeg-quality", type=int, default=70, help="MJPEG quality from 1 to 100.")
    parser.add_argument(
        "--stream-annotated",
        action="store_true",
        help="Stream the annotated tracking view instead of raw RGB frames.",
    )
    parser.add_argument(
        "--stream-camera-timeout-seconds",
        type=float,
        default=3.0,
        help="Seconds without a published frame before a camera is reported offline.",
    )
    parser.add_argument("--disable-streaming", action="store_true", help="Disable the MJPEG streaming API.")
    parser.add_argument(
        "--hide-depth-window",
        action="store_true",
        default=True,
        help="Show only the synchronized RGB window. This is the default.",
    )
    parser.add_argument(
        "--show-depth-window",
        action="store_false",
        dest="hide_depth_window",
        help="Show the synchronized tiled depth window for debugging.",
    )
    parser.add_argument("--max-window-width", type=int, default=1600, help="Maximum RGB/depth window width.")
    parser.add_argument("--max-window-height", type=int, default=900, help="Maximum RGB/depth window height.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional live artifact output directory.")
    parser.add_argument(
        "--state-db",
        type=Path,
        default=DEFAULT_SHOP_STATE_DB,
        help="SQLite database for operational live visit state. Default: state/shop_state.sqlite.",
    )
    parser.add_argument("--detector-backend", choices=DETECTOR_BACKEND_CHOICES, default=DEFAULT_PERSON_DETECTOR_BACKEND)
    parser.add_argument("--model", type=Path, default=DEFAULT_PERSON_DETECTOR_MODEL)
    parser.add_argument("--input-width", type=int, default=DEFAULT_DETECTION_INPUT_WIDTH)
    parser.add_argument("--input-height", type=int, default=DEFAULT_DETECTION_INPUT_HEIGHT)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD)
    parser.add_argument("--nms-threshold", type=float, default=DEFAULT_DETECTION_NMS_THRESHOLD)
    parser.add_argument("--yolo-person-class-id", type=int, default=0)
    parser.add_argument("--tracker-backend", choices=["iou"], default=DEFAULT_PERSON_TRACKER_BACKEND)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_TRACKING_IOU_THRESHOLD)
    parser.add_argument("--max-missed", type=int, default=DEFAULT_TRACKING_MAX_MISSED)
    parser.add_argument(
        "--plane-json",
        type=Path,
        default=None,
        help="Optional explicit plane-fit JSON to use for every entrance-capable stream.",
    )
    parser.add_argument("--depth-trigger-mode", choices=["threshold", "plane"], default="plane")
    parser.add_argument("--depth-threshold-mm", type=int, default=2000)
    parser.add_argument("--depth-hysteresis-mm", type=int, default=250)
    parser.add_argument("--plane-point-x-mm", type=float, default=0.0)
    parser.add_argument("--plane-point-y-mm", type=float, default=0.0)
    parser.add_argument("--plane-point-z-mm", type=float, default=2000.0)
    parser.add_argument("--plane-normal-x", type=float, default=0.0)
    parser.add_argument("--plane-normal-y", type=float, default=0.0)
    parser.add_argument("--plane-normal-z", type=float, default=1.0)
    parser.add_argument(
        "--plane-enter-direction",
        choices=["positive_to_negative", "negative_to_positive"],
        default="positive_to_negative",
    )
    parser.add_argument("--plane-hysteresis-mm", type=float, default=150.0)
    parser.add_argument("--depth-min-valid-pixels", type=int, default=25)
    parser.add_argument("--depth-roi-width-fraction", type=float, default=0.30)
    parser.add_argument("--depth-roi-height-fraction", type=float, default=0.22)
    parser.add_argument("--shop-api-base-url", type=str, default=None, help="Optional shop API base URL.")
    parser.add_argument("--shop-api-key", type=str, default=None, help="Shop API X-Api-Key value.")
    parser.add_argument("--shop-id", type=int, default=None, help="Shop id for shop API visit binding.")
    parser.add_argument("--shop-api-max-age-seconds", type=int, default=30)
    parser.add_argument("--shop-api-timeout-seconds", type=float, default=2.0)
    parser.add_argument(
        "--log-plane-trace",
        action="store_true",
        help=(
            "Log per-frame plane signed distance for each entrance-capable camera track. "
            "Useful for debugging missed entry/leave crossings and track splits."
        ),
    )
    add_plane_track_split_recovery_args(parser)
    add_face_identity_args(parser)
    add_body_evidence_args(parser)
    add_visit_identity_args(parser)
    add_visit_registry_args(parser)
    return parser


def resolve_camera_roles(args: argparse.Namespace) -> list[str]:
    if args.camera_role is None or len(args.camera_role) == 0:
        return [CAMERA_ROLE_ENTRANCE for _device_id in args.device_id]
    if len(args.camera_role) != len(args.device_id):
        raise ValueError("--camera-role must be omitted or provide exactly one role per --device-id.")
    return list(args.camera_role)


def resolve_live_device(device_id: str) -> dai.Device:
    available = list_available_devices()
    matching = [info for info in available if device_identifier(info) == device_id]
    if not matching:
        available_ids = ", ".join(device_identifier(info) for info in available) or "none"
        raise RuntimeError(f"Requested device-id '{device_id}' not found. Available device ids: {available_ids}")
    device = dai.Device(device_id)
    configure_live_device(device)
    return device


def choose_best_depth_packet(
    packets: deque[LiveDepthPacket],
    rgb_host_synced_seconds: float,
) -> LiveDepthPacket | None:
    if not packets:
        return None
    return min(packets, key=lambda packet: abs(packet.host_synced_seconds - rgb_host_synced_seconds))


def create_live_stream_state(
    *,
    device_id: str,
    camera_role: str,
    args: argparse.Namespace,
    stack: ExitStack,
) -> LiveSyncedStreamState:
    device = resolve_live_device(device_id)
    calibration = device.readCalibration()
    intrinsics = intrinsics_from_matrix(
        calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, (args.frame_width, args.frame_height))
    )

    pipeline = stack.enter_context(dai.Pipeline(device))
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_output = cam_rgb.requestOutput(
        size=(args.frame_width, args.frame_height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=args.fps,
    )

    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(args.frame_width, args.frame_height)
    mono_left.requestFullResolutionOutput(fps=args.fps).link(stereo.left)
    mono_right.requestFullResolutionOutput(fps=args.fps).link(stereo.right)

    rgb_queue = rgb_output.createOutputQueue(maxSize=4, blocking=False)
    depth_queue = stereo.depth.createOutputQueue(maxSize=8, blocking=False)
    tracker = build_person_tracker(args)
    state = LiveSyncedStreamState(
        device_id=device_id,
        camera_role=camera_role,
        device=device,
        pipeline=pipeline,
        rgb_queue=rgb_queue,
        depth_queue=depth_queue,
        intrinsics=intrinsics,
        tracker=tracker,
    )

    if args.depth_trigger_mode == "plane" and is_entrance_enabled(camera_role):
        args_for_stream = argparse.Namespace(**vars(args))
        args_for_stream.plane_json = resolve_plane_json_path(
            plane_json=args.plane_json,
            device_id=device_id,
            calibrations_root=DEFAULT_PLANE_CALIBRATIONS_DIR,
            recording_dir=None,
        )
        if args_for_stream.plane_json is None:
            raise FileNotFoundError(
                "Plane mode requested, but no plane JSON was provided and "
                f"no calibration was found for device {device_id}."
            )
        state.plane = plane_from_args(args_for_stream)
        state.plane_enter_direction = plane_enter_direction_from_args(args_for_stream)
        print(f"Loaded plane for {device_id} from {args_for_stream.plane_json}")
    elif args.depth_trigger_mode == "plane":
        print(f"Skipping plane load for observer-only device {device_id}.")

    pipeline.start()
    print(f"Started live RGBD stream device_id={device_id} role={camera_role}")
    return state


def write_live_config(
    *,
    artifact_writer: ReplayArtifactWriter,
    args: argparse.Namespace,
    camera_roles: list[str],
) -> None:
    if artifact_writer.output_dir is None:
        return
    payload = {
        "type": "live_synced_rgbd_streams_config",
        "device_ids": list(args.device_id),
        "camera_roles": camera_roles,
        "width": args.frame_width,
        "height": args.frame_height,
        "fps": args.fps,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "shop_api_key"
        },
    }
    (artifact_writer.output_dir / "live_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def drain_depth_queue(state: LiveSyncedStreamState) -> None:
    depth_msg = state.depth_queue.tryGet()
    while depth_msg is not None:
        frame_mm = depth_msg.getFrame().copy()
        state.recent_depth_packets.append(
            LiveDepthPacket(
                sequence_num=int(depth_msg.getSequenceNum()),
                host_synced_seconds=float(depth_msg.getTimestamp().total_seconds()),
                device_monotonic_seconds=float(depth_msg.getTimestampDevice().total_seconds()),
                frame_mm=frame_mm,
            )
        )
        state.latest_depth_visual = colorize_depth(frame_mm)
        depth_msg = state.depth_queue.tryGet()


def process_latest_rgb_frame(
    *,
    camera_index: int,
    state: LiveSyncedStreamState,
    detector: PersonDetector,
    face_matcher: FaceRecognizer | None,
    body_evidence_extractor: BodyEvidenceExtractor,
    visit_registry: VisitRegistry,
    artifact_writer: ReplayArtifactWriter,
    shop_api_client: ShopApiClient,
    shop_state_store: ShopStateStore,
    customer_ids_by_visit: dict[int, str],
    args: argparse.Namespace,
) -> None:
    rgb_msg = state.rgb_queue.tryGet()
    if rgb_msg is None:
        return

    rgb_sequence = int(rgb_msg.getSequenceNum())
    if state.last_processed_rgb_sequence == rgb_sequence:
        return

    rgb_host_synced_seconds = float(rgb_msg.getTimestamp().total_seconds())
    best_depth = choose_best_depth_packet(state.recent_depth_packets, rgb_host_synced_seconds)
    rgb_frame = rgb_msg.getCvFrame()
    state.cached_rgb_raw = rgb_frame
    state.last_rgb_host_synced_seconds = rgb_host_synced_seconds
    state.last_processed_rgb_sequence = rgb_sequence

    if best_depth is None:
        overlay = rgb_frame.copy()
        cv2.putText(
            overlay,
            "Waiting for aligned depth...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        state.cached_rgb_overlay = overlay
        state.cached_observer_snapshot = None
        return

    processed_frame = build_processed_live_rgb_frame(
        camera_index=camera_index,
        state=state,
        rgb_frame=rgb_frame,
        depth_frame_mm=best_depth.frame_mm,
        rgb_host_synced_seconds=rgb_host_synced_seconds,
        rgb_sequence_num=rgb_sequence,
        depth_packet=best_depth,
        detector=detector,
        face_matcher=face_matcher,
        body_evidence_extractor=body_evidence_extractor,
        visit_registry=visit_registry,
        artifact_writer=artifact_writer,
        shop_api_client=shop_api_client,
        shop_state_store=shop_state_store,
        customer_ids_by_visit=customer_ids_by_visit,
        args=args,
    )
    state.cached_rgb_overlay = processed_frame.overlay
    state.cached_observer_snapshot = processed_frame.observer_snapshot
    state.cached_depth_overlay = colorize_depth(best_depth.frame_mm)


def build_processed_live_rgb_frame(
    *,
    camera_index: int,
    state: LiveSyncedStreamState,
    rgb_frame: np.ndarray,
    depth_frame_mm: np.ndarray,
    rgb_host_synced_seconds: float,
    rgb_sequence_num: int,
    depth_packet: LiveDepthPacket,
    detector: PersonDetector,
    face_matcher: FaceRecognizer | None,
    body_evidence_extractor: BodyEvidenceExtractor,
    visit_registry: VisitRegistry,
    artifact_writer: ReplayArtifactWriter,
    shop_api_client: ShopApiClient,
    shop_state_store: ShopStateStore,
    customer_ids_by_visit: dict[int, str],
    args: argparse.Namespace,
) -> ProcessedLiveFrame:
    detections = detector.detect(rgb_frame)
    tracks = state.tracker.update(detections)
    entrance_enabled = is_entrance_enabled(state.camera_role)

    if args.depth_trigger_mode == "plane" and entrance_enabled:
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
            host_seconds=rgb_host_synced_seconds,
            track_split_recovery=args.plane_track_split_recovery,
            track_split_recovery_max_age_seconds=args.plane_track_split_recovery_max_age_seconds,
            track_split_recovery_max_centroid_distance_px=args.plane_track_split_recovery_max_centroid_distance_px,
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
    exited_track_ids = depth_result.exited_track_ids
    depth_samples = depth_result.depth_samples
    signed_distances_mm = depth_result.signed_distances_mm
    entry_reasons_by_track = depth_result.entry_reasons_by_track
    recovered_entry_source_track_ids = depth_result.recovered_entry_source_track_ids
    leave_reasons_by_track = depth_result.leave_reasons_by_track
    recovered_leave_source_track_ids = depth_result.recovered_leave_source_track_ids
    if not entrance_enabled:
        entered_track_ids = []
        exited_track_ids = []
    if args.log_plane_trace and args.depth_trigger_mode == "plane" and entrance_enabled:
        log_plane_trace(
            prefix="LIVE_PLANE_TRACE",
            device_id=state.device_id,
            host_seconds=rgb_host_synced_seconds,
            tracks=tracks,
            depth_samples=depth_samples,
            signed_distances_mm=signed_distances_mm,
            depth_states=state.depth_states,
            entered_track_ids=set(entered_track_ids),
            exited_track_ids=set(exited_track_ids),
        )

    recognized_faces = []
    if face_matcher is not None:
        face_tracks = face_recognition_eligible_tracks(
            tracks,
            min_width_px=args.face_min_track_width_px,
            min_height_px=args.face_min_track_height_px,
        )
        recognized_faces = face_matcher.recognize(rgb_frame, tracks=face_tracks)

    body_evidence_by_track = body_evidence_extractor.extract(rgb_frame, tracks=tracks)
    frame_evidence = FrameEvidence(
        device_id=state.device_id,
        host_seconds=rgb_host_synced_seconds,
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

    entered_visit_ids_this_frame: set[int] = set()
    for track_id in entered_track_ids:
        sample = depth_samples.get(track_id)
        if sample is None:
            continue
        track_evidence = track_visit_evidence_by_id.get(track_id)
        if track_evidence is not None:
            decision = visit_registry.resolve_entrance_track(track_evidence)
            visit_assignments[track_id] = decision.assignment
            artifact_writer.write_visit_decision(
                resolution="entrance_track",
                track_evidence=track_evidence,
                decision=decision,
            )

        visit_assignment = visit_assignments.get(track_id)
        visit_id = None if visit_assignment is None else visit_assignment.visit_id
        if visit_assignment is not None:
            entered_visit_ids_this_frame.add(visit_assignment.visit_id)
        event_payload = {
            "type": "live_depth_plane_entry_event"
            if args.depth_trigger_mode == "plane"
            else "live_depth_entry_event",
            "device_id": state.device_id,
            "camera_role": state.camera_role,
            "track_id": track_id,
            "visit_id": visit_id,
            "host_synced_seconds": rgb_host_synced_seconds,
            "rgb_sequence_num": rgb_sequence_num,
            "depth_sequence_num": depth_packet.sequence_num,
            "matched_depth_delta_ms": (depth_packet.host_synced_seconds - rgb_host_synced_seconds) * 1000.0,
            "depth_mm": sample.depth_mm,
            "plane_signed_distance_mm": signed_distances_mm.get(track_id)
            if args.depth_trigger_mode == "plane"
            else None,
            "entry_reason": entry_reasons_by_track.get(track_id, "direct_crossing"),
            "recovered_entry_source_track_id": recovered_entry_source_track_ids.get(track_id),
        }
        artifact_writer.write_entrance_event(event_payload)
        shop_state_store.record_entry(
            visit_id=visit_id,
            host_seconds=rgb_host_synced_seconds,
            device_id=state.device_id,
            camera_role=state.camera_role,
            track_id=track_id,
            depth_mm=sample.depth_mm,
            plane_signed_distance_mm=signed_distances_mm.get(track_id)
            if args.depth_trigger_mode == "plane"
            else None,
            reason=entry_reasons_by_track.get(track_id, "direct_crossing"),
            event_payload=event_payload,
        )
        if args.depth_trigger_mode == "plane":
            print(
                f"LIVE_DEPTH_PLANE_ENTRY_EVENT device_id={state.device_id} "
                f"track_id={track_id} visit_id={visit_id} "
                f"reason={entry_reasons_by_track.get(track_id, 'direct_crossing')} "
                f"source_track_id={recovered_entry_source_track_ids.get(track_id)} "
                f"host_synced_seconds={rgb_host_synced_seconds:.3f} "
                f"plane_mm={signed_distances_mm.get(track_id, float('nan')):.0f} "
                f"depth_mm={sample.depth_mm:.0f}"
            )
            if visit_assignment is not None:
                visit_plane_state = state.visit_plane_states.setdefault(
                    visit_assignment.visit_id,
                    VisitPlaneState(),
                )
                visit_plane_state.entered = True
                visit_plane_state.inside_track_ids_after_entry = {track_id}
                visit_plane_state.last_signed_distance_mm = signed_distances_mm.get(track_id)
                visit_plane_state.last_track_id = track_id
                visit_plane_state.last_seen_seconds = rgb_host_synced_seconds
        else:
            print(
                f"LIVE_DEPTH_ENTRY_EVENT device_id={state.device_id} "
                f"track_id={track_id} visit_id={visit_id} "
                f"host_synced_seconds={rgb_host_synced_seconds:.3f} depth_mm={sample.depth_mm:.0f}"
            )
        try:
            shopping_customer_id = shop_api_client.bind_visit(visit_id)
            shop_state_store.record_shop_customer_binding(
                visit_id=visit_id,
                shopping_customer_id=shopping_customer_id,
            )
            if visit_id is not None and shopping_customer_id is not None:
                customer_ids_by_visit[visit_id] = shopping_customer_id
        except RuntimeError as exc:
            print(f"SHOP_API_BIND_ERROR visit_id={visit_id} error={exc}")

    visit_plane_leave_track_ids: list[int] = []
    if args.depth_trigger_mode == "plane" and entrance_enabled:
        for track_id, visit_assignment in visit_assignments.items():
            if track_id in entered_track_ids or track_id in exited_track_ids:
                continue
            if visit_assignment.visit_id in entered_visit_ids_this_frame:
                continue
            signed_distance_mm = signed_distances_mm.get(track_id)
            if signed_distance_mm is None:
                continue
            visit_plane_state = state.visit_plane_states.setdefault(
                visit_assignment.visit_id,
                VisitPlaneState(),
            )
            if (
                visit_plane_state.entered
                and track_id in visit_plane_state.inside_track_ids_after_entry
                and _visit_plane_outside(
                    signed_distance_mm,
                    plane_enter_direction=str(state.plane_enter_direction),
                    plane_hysteresis_mm=float(args.plane_hysteresis_mm),
                )
            ):
                visit_plane_state.entered = False
                visit_plane_state.inside_track_ids_after_entry.clear()
                visit_plane_state.last_signed_distance_mm = signed_distance_mm
                visit_plane_state.last_track_id = track_id
                visit_plane_state.last_seen_seconds = rgb_host_synced_seconds
                visit_plane_leave_track_ids.append(track_id)
                leave_reasons_by_track[track_id] = "visit_plane_crossing"
            elif _visit_plane_inside(
                signed_distance_mm,
                plane_enter_direction=str(state.plane_enter_direction),
            ):
                visit_plane_state.inside_track_ids_after_entry.add(track_id)
                visit_plane_state.last_signed_distance_mm = signed_distance_mm
                visit_plane_state.last_track_id = track_id
                visit_plane_state.last_seen_seconds = rgb_host_synced_seconds

    closed_visit_ids_this_frame: set[int] = set()
    for track_id in [*exited_track_ids, *visit_plane_leave_track_ids]:
        sample = depth_samples.get(track_id)
        if sample is None:
            continue
        visit_assignment = visit_assignments.get(track_id)
        recovered_source_track_id = recovered_leave_source_track_ids.get(track_id)
        if visit_assignment is None and recovered_source_track_id is not None:
            visit_assignment = visit_registry.assignment_for_track(
                device_id=state.device_id,
                track_id=recovered_source_track_id,
            )
        visit_id = None if visit_assignment is None else visit_assignment.visit_id
        if visit_id is not None and visit_id in closed_visit_ids_this_frame:
            continue
        event_payload = {
            "type": "live_depth_plane_leave_event"
            if args.depth_trigger_mode == "plane"
            else "live_depth_leave_event",
            "device_id": state.device_id,
            "camera_role": state.camera_role,
            "track_id": track_id,
            "visit_id": visit_id,
            "host_synced_seconds": rgb_host_synced_seconds,
            "rgb_sequence_num": rgb_sequence_num,
            "depth_sequence_num": depth_packet.sequence_num,
            "matched_depth_delta_ms": (depth_packet.host_synced_seconds - rgb_host_synced_seconds) * 1000.0,
            "depth_mm": sample.depth_mm,
            "plane_signed_distance_mm": signed_distances_mm.get(track_id)
            if args.depth_trigger_mode == "plane"
            else None,
            "leave_reason": leave_reasons_by_track.get(track_id, "direct_crossing"),
            "recovered_leave_source_track_id": recovered_source_track_id,
        }
        artifact_writer.write_entrance_event(event_payload)
        shop_state_store.record_leave(
            visit_id=visit_id,
            host_seconds=rgb_host_synced_seconds,
            device_id=state.device_id,
            camera_role=state.camera_role,
            track_id=track_id,
            depth_mm=sample.depth_mm,
            plane_signed_distance_mm=signed_distances_mm.get(track_id)
            if args.depth_trigger_mode == "plane"
            else None,
            reason=leave_reasons_by_track.get(track_id, "direct_crossing"),
            event_payload=event_payload,
        )
        visit_registry.close_visit(visit_id, host_seconds=rgb_host_synced_seconds)
        if visit_id is not None:
            closed_visit_ids_this_frame.add(visit_id)
        if args.depth_trigger_mode == "plane":
            if visit_assignment is not None:
                visit_plane_state = state.visit_plane_states.setdefault(
                    visit_assignment.visit_id,
                    VisitPlaneState(),
                )
                visit_plane_state.entered = False
                visit_plane_state.inside_track_ids_after_entry.clear()
                visit_plane_state.last_signed_distance_mm = signed_distances_mm.get(track_id)
                visit_plane_state.last_track_id = track_id
                visit_plane_state.last_seen_seconds = rgb_host_synced_seconds
            print(
                f"LIVE_DEPTH_PLANE_LEAVE_EVENT device_id={state.device_id} "
                f"track_id={track_id} visit_id={visit_id} "
                f"reason={leave_reasons_by_track.get(track_id, 'direct_crossing')} "
                f"source_track_id={recovered_source_track_id} "
                f"host_synced_seconds={rgb_host_synced_seconds:.3f} "
                f"plane_mm={signed_distances_mm.get(track_id, float('nan')):.0f} "
                f"depth_mm={sample.depth_mm:.0f}"
            )
        else:
            print(
                f"LIVE_DEPTH_LEAVE_EVENT device_id={state.device_id} "
                f"track_id={track_id} visit_id={visit_id} "
                f"host_synced_seconds={rgb_host_synced_seconds:.3f} depth_mm={sample.depth_mm:.0f}"
            )
        try:
            shop_api_client.mark_left(visit_id)
        except RuntimeError as exc:
            print(f"SHOP_API_MARK_LEFT_ERROR visit_id={visit_id} error={exc}")

    observer_snapshot = None
    if observer_enabled:
        observer_snapshot = build_observer_camera_snapshot(
            camera_index=camera_index,
            device_id=state.device_id,
            camera_role=state.camera_role,
            rgb_frame=rgb_frame,
            rgb_sequence_number=rgb_sequence_num,
            host_synced_seconds=rgb_host_synced_seconds,
            tracks=tracks,
            track_visit_evidence_by_id=track_visit_evidence_by_id,
            visit_assignments=visit_assignments,
            depth_samples=depth_samples,
            customer_ids_by_visit=customer_ids_by_visit,
        )

    overlay = rgb_frame.copy()
    draw_tracks(overlay, tracks)
    draw_visit_labels(overlay, tracks, visit_assignments, show_face_evidence=False)
    draw_depth_samples(
        overlay,
        tracks=tracks,
        depth_samples=depth_samples,
        depth_threshold_mm=float(args.depth_threshold_mm),
        signed_distances_mm=signed_distances_mm,
        plane_mode=args.depth_trigger_mode == "plane",
    )
    if face_matcher is not None:
        draw_recognized_faces(overlay, recognized_faces, show_labels=False)
    return ProcessedLiveFrame(overlay=overlay, observer_snapshot=observer_snapshot)


def placeholder_frame(label: str, *, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_devices:
        print_available_devices()
        return
    if not args.device_id:
        raise ValueError("--device-id is required unless --list-devices is set.")
    if not 1 <= args.stream_port <= 65535:
        raise ValueError("--stream-port must be between 1 and 65535.")
    if not 1 <= args.stream_jpeg_quality <= 100:
        raise ValueError("--stream-jpeg-quality must be between 1 and 100.")
    if args.stream_camera_timeout_seconds <= 0:
        raise ValueError("--stream-camera-timeout-seconds must be greater than zero.")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise ValueError("--frame-width and --frame-height must be greater than zero.")

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    camera_roles = resolve_camera_roles(args)
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
        observer_handoff_min_delay_seconds=args.observer_handoff_min_delay_seconds,
        observer_handoff_max_delay_seconds=args.observer_handoff_max_delay_seconds,
        observer_handoff_threshold=args.observer_handoff_threshold,
        log_decisions=args.log_visit_decisions,
    )
    shop_state_store = ShopStateStore(args.state_db)
    customer_ids_by_visit = shop_state_store.load_shop_customer_bindings()
    visit_registry.next_visit_id = max(visit_registry.next_visit_id, shop_state_store.next_visit_id())
    print(
        f"Using shop state DB {shop_state_store.db_path} "
        f"next_visit_id={visit_registry.next_visit_id}"
    )
    shop_api_client = ShopApiClient(
        base_url=args.shop_api_base_url,
        api_key=args.shop_api_key,
        shop_id=args.shop_id,
        max_age_seconds=args.shop_api_max_age_seconds,
        timeout_seconds=args.shop_api_timeout_seconds,
    )
    artifact_writer = ReplayArtifactWriter(args.output_dir)
    write_live_config(artifact_writer=artifact_writer, args=args, camera_roles=camera_roles)
    if artifact_writer.enabled:
        print(f"Writing live artifacts to {artifact_writer.output_dir}")

    stream_server: MjpegStreamServer | None = None
    states: list[LiveSyncedStreamState] = []
    try:
        if not args.disable_streaming:
            stream_server = MjpegStreamServer(
                camera_device_ids=list(args.device_id),
                camera_roles=camera_roles,
                host=args.stream_host,
                port=args.stream_port,
                jpeg_quality=args.stream_jpeg_quality,
                camera_timeout_seconds=args.stream_camera_timeout_seconds,
            )
            stream_server.start()

        with ExitStack() as stack:
            states = [
                create_live_stream_state(
                    device_id=device_id,
                    camera_role=camera_role,
                    args=args,
                    stack=stack,
                )
                for device_id, camera_role in zip(args.device_id, camera_roles)
            ]
            print(
                "Camera roles: "
                + ", ".join(f"{state.device_id}={state.camera_role}" for state in states)
            )
            if args.headless:
                print("Running headless: OpenCV windows are disabled.")
            else:
                print("Controls: q=quit")
                cv2.namedWindow("Live Synchronized RGBD - RGB", cv2.WINDOW_NORMAL)
                if not args.hide_depth_window:
                    cv2.namedWindow("Live Synchronized RGBD - Depth", cv2.WINDOW_NORMAL)

            while not stop_requested.is_set():
                for camera_index, state in enumerate(states):
                    if state.device.isClosed() or not state.pipeline.isRunning():
                        raise RuntimeError(f"Live pipeline stopped for device {state.device_id}.")
                    drain_depth_queue(state)
                    previous_rgb_sequence = state.last_processed_rgb_sequence
                    process_latest_rgb_frame(
                        camera_index=camera_index,
                        state=state,
                        detector=detector,
                        face_matcher=face_matcher,
                        body_evidence_extractor=body_evidence_extractor,
                        visit_registry=visit_registry,
                        artifact_writer=artifact_writer,
                        shop_api_client=shop_api_client,
                        shop_state_store=shop_state_store,
                        customer_ids_by_visit=customer_ids_by_visit,
                        args=args,
                    )
                    if (
                        stream_server is not None
                        and state.last_processed_rgb_sequence != previous_rgb_sequence
                    ):
                        stream_frame = (
                            state.cached_rgb_overlay
                            if args.stream_annotated
                            else state.cached_rgb_raw
                        )
                        if stream_frame is not None:
                            stream_server.publish(camera_index, stream_frame)
                        if state.cached_observer_snapshot is not None:
                            stream_server.publish_observer_snapshot(
                                camera_index,
                                state.cached_observer_snapshot,
                            )

                if not args.headless:
                    rgb_frames = [
                        state.cached_rgb_overlay
                        if state.cached_rgb_overlay is not None
                        else placeholder_frame(
                            f"Camera {index + 1}: waiting",
                            width=args.frame_width,
                            height=args.frame_height,
                        )
                        for index, state in enumerate(states)
                    ]
                    rgb_grid = tile_frames([frame.copy() for frame in rgb_frames], args.columns)
                    cv2.imshow(
                        "Live Synchronized RGBD - RGB",
                        fit_to_window(
                            rgb_grid,
                            max_width=args.max_window_width,
                            max_height=args.max_window_height,
                        ),
                    )

                    if not args.hide_depth_window:
                        depth_frames = [
                            state.cached_depth_overlay
                            if state.cached_depth_overlay is not None
                            else placeholder_frame(
                                f"Camera {index + 1}: no depth",
                                width=args.frame_width,
                                height=args.frame_height,
                            )
                            for index, state in enumerate(states)
                        ]
                        depth_grid = tile_frames([frame.copy() for frame in depth_frames], args.columns)
                        cv2.imshow(
                            "Live Synchronized RGBD - Depth",
                            fit_to_window(
                                depth_grid,
                                max_width=args.max_window_width,
                                max_height=args.max_window_height,
                            ),
                        )

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if stream_server is not None:
            stream_server.stop()
        artifact_writer.write_final_visits(visit_registry)
        artifact_writer.close()
        shop_state_store.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
