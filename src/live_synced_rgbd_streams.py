import argparse
import json
import math
import signal
import subprocess
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

from pipeline.body_evidence import (
    BodyEvidence,
    BodyEvidenceExtractor,
    add_body_evidence_args,
    build_body_evidence_extractor,
    scale_body_evidence_heights,
)
from pipeline.camera import (
    configure_live_device,
    device_identifier,
    format_usb_connection,
    list_available_devices,
    print_available_devices,
)
from pipeline.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_PERSON_DETECTOR_BACKEND,
    DEFAULT_PERSON_DETECTOR_MODEL,
    DEFAULT_PERSON_TRACKER_BACKEND,
    DEFAULT_MAX_RGB_DEPTH_DELTA_MS,
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
)
from pipeline.depth import (
    CameraIntrinsics,
    DepthSample,
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
    scale_depth_samples,
)
from pipeline.detection import DETECTOR_BACKEND_CHOICES, PersonDetector, build_person_detector
from pipeline.face_identity import (
    FaceRecognizer,
    RecognizedFace,
    add_face_identity_args,
    build_face_recognizer,
    draw_recognized_faces,
    face_recognition_eligible_tracks,
    scale_recognized_faces,
)
from pipeline.mjpeg_stream_server import MjpegStreamServer
from pipeline.observer_api import ObserverCameraSnapshot, build_observer_camera_snapshot
from pipeline.performance import LivePerformanceLogger
from pipeline.rgbd_recording import DEFAULT_PLANE_CALIBRATIONS_DIR
from pipeline.shelf_anchors import (
    ShelfAnchor,
    ShelfAnchorManager,
)
from pipeline.shelf_api import (
    ShelfCameraSnapshot,
    build_shelf_camera_snapshot,
    shelf_event_payload,
)
from pipeline.shelf_config import (
    DEFAULT_SHELF_CALIBRATIONS_DIR,
    DEFAULT_SHELF_CONFIG_PATH,
    ShelfWatchingConfig,
    load_shelf_config,
    validate_shelf_config_for_live_cameras,
)
from pipeline.shelf_person_depth import sample_shelf_person_depth
from pipeline.shelf_proximity import (
    ShelfCameraObservation,
    ShelfProximityCoordinator,
    ShelfProximityEvent,
    person_to_shelf_distance_mm,
)
from pipeline.shop_state_store import DEFAULT_SHOP_STATE_DB, ShopStateStore
from pipeline.tracking import PersonTracker, Track, build_person_tracker, draw_tracks, scale_tracks
from pipeline.visit_identity import VisitAssignment, add_visit_identity_args, draw_visit_labels
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


DEFAULT_LIVE_FRAME_WIDTH = 1920
DEFAULT_LIVE_FRAME_HEIGHT = 1080
DEFAULT_LIVE_PROCESSING_WIDTH = 1280
DEFAULT_LIVE_PROCESSING_HEIGHT = 720


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
    shelf_snapshot: ShelfCameraSnapshot | None
    shelf_events: tuple[ShelfProximityEvent, ...]


@dataclass(frozen=True)
class LiveRgbTrackSnapshot:
    sequence_num: int
    host_synced_seconds: float
    processing_frame: np.ndarray
    processing_tracks: tuple[Track, ...]
    display_tracks: tuple[Track, ...]
    recognized_faces: tuple[RecognizedFace, ...]
    body_evidence_by_track: dict[int, BodyEvidence]


@dataclass
class LiveSyncedStreamState:
    device_id: str
    camera_role: str
    device: dai.Device
    pipeline: dai.Pipeline
    rgb_queue: dai.MessageQueue
    processing_rgb_queue: dai.MessageQueue
    depth_queue: dai.MessageQueue
    intrinsics: CameraIntrinsics
    tracker: PersonTracker
    depth_states: dict[int, DepthEntranceState] = field(default_factory=dict)
    visit_plane_states: dict[int, VisitPlaneState] = field(default_factory=dict)
    recent_raw_rgb_messages: deque[Any] = field(default_factory=deque)
    recent_processing_rgb_messages: deque[Any] = field(default_factory=deque)
    recent_rgb_track_snapshots: deque[LiveRgbTrackSnapshot] = field(default_factory=deque)
    plane: object | None = None
    plane_enter_direction: str | None = None
    cached_rgb_raw: np.ndarray | None = None
    cached_rgb_overlay: np.ndarray | None = None
    cached_observer_snapshot: ObserverCameraSnapshot | None = None
    cached_shelf_snapshot: ShelfCameraSnapshot | None = None
    pending_shelf_events: list[ShelfProximityEvent] = field(default_factory=list)
    cached_depth_overlay: np.ndarray | None = None
    last_raw_rgb_sequence: int | None = None
    last_rgb_evidence_sequence: int | None = None
    last_processed_rgb_sequence: int | None = None
    last_raw_rgb_host_synced_seconds: float | None = None
    last_rgb_host_synced_seconds: float | None = None
    last_depth_sync_warning_seconds: float | None = None
    shelf_anchor_manager: ShelfAnchorManager | None = None


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
        default=DEFAULT_LIVE_FRAME_WIDTH,
        help="RGB camera and raw MJPEG stream width. Default: 1920.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_LIVE_FRAME_HEIGHT,
        help="RGB camera and raw MJPEG stream height. Default: 1080.",
    )
    parser.add_argument(
        "--processing-width",
        type=int,
        default=DEFAULT_LIVE_PROCESSING_WIDTH,
        help="Width used for detection, tracking, and aligned depth. Default: 1280.",
    )
    parser.add_argument(
        "--processing-height",
        type=int,
        default=DEFAULT_LIVE_PROCESSING_HEIGHT,
        help="Height used for detection, tracking, and aligned depth. Default: 720.",
    )
    parser.add_argument(
        "--depth-median-filter",
        choices=("off", "3x3", "5x5", "7x7"),
        default="7x7",
        help=(
            "StereoDepth on-device median filter kernel. Default: 7x7. "
            "Use off to disable spatial smoothing."
        ),
    )
    parser.add_argument(
        "--max-rgb-depth-delta-ms",
        type=float,
        default=DEFAULT_MAX_RGB_DEPTH_DELTA_MS,
        help=(
            "Reject RGB/depth pairs farther apart than this before running depth or visit logic. "
            "Default: 250 ms."
        ),
    )
    parser.add_argument(
        "--processing-buffer-seconds",
        type=float,
        default=6.0,
        help=(
            "Seconds of low-resolution track/evidence snapshots retained while waiting for delayed aligned depth. "
            "Default: 6."
        ),
    )
    parser.add_argument("--columns", type=int, default=2, help="Columns for tiled live view.")
    parser.add_argument(
        "--show-annotated-preview",
        action="store_true",
        default=True,
        help="Show the delayed synchronized processing overlay in the GUI. This is the default.",
    )
    parser.add_argument(
        "--show-raw-preview",
        action="store_false",
        dest="show_annotated_preview",
        help="Show current raw RGB in the GUI instead of the delayed annotated overlay.",
    )
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
        "--enable-operator-console",
        action="store_true",
        help=(
            "Enable the mobile operator test console and test-run API. "
            "Requires --operator-api-token."
        ),
    )
    parser.add_argument(
        "--operator-runs-root",
        type=Path,
        default=Path("test-runs"),
        help="Directory for exported operator test runs. Default: test-runs.",
    )
    parser.add_argument(
        "--operator-api-token",
        help=(
            "Bearer token for the operator console and test-run API. "
            "Used only with --enable-operator-console."
        ),
    )
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
    parser.add_argument(
        "--log-performance",
        action="store_true",
        help="Log aggregated live-pipeline stage timings for performance analysis.",
    )
    parser.add_argument(
        "--performance-log-interval-seconds",
        type=float,
        default=5.0,
        help="Aggregation window for --log-performance. Default: 5 seconds.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional live artifact output directory.")
    parser.add_argument(
        "--state-db",
        type=Path,
        default=DEFAULT_SHOP_STATE_DB,
        help="SQLite database for operational live visit state. Default: state/shop_state.sqlite.",
    )
    parser.add_argument(
        "--enable-shelf-watching",
        action="store_true",
        help="Enable shelf distance watching for cameras with saved shelf anchors.",
    )
    parser.add_argument(
        "--shelf-config",
        type=Path,
        default=DEFAULT_SHELF_CONFIG_PATH,
        help="Shelf catalog and proximity thresholds. Default: config/shelves.json.",
    )
    parser.add_argument(
        "--shelf-calibrations-root",
        type=Path,
        default=DEFAULT_SHELF_CALIBRATIONS_DIR,
        help="Directory containing per-camera shelf anchor calibrations.",
    )
    parser.add_argument(
        "--log-shelf-distance",
        action="store_true",
        help="Log detailed shelf person-distance diagnostics.",
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


def validate_operator_console_args(args: argparse.Namespace) -> None:
    if args.enable_operator_console and not args.operator_api_token:
        raise ValueError(
            "--operator-api-token is required with --enable-operator-console."
        )
    if args.operator_api_token and not args.enable_operator_console:
        raise ValueError(
            "--operator-api-token requires --enable-operator-console."
        )
    if args.enable_operator_console and args.disable_streaming:
        raise ValueError(
            "--enable-operator-console cannot be used with --disable-streaming."
        )


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


def drain_latest_message(queue: dai.MessageQueue) -> Any | None:
    latest_message = queue.tryGet()
    if latest_message is None:
        return None

    next_message = queue.tryGet()
    while next_message is not None:
        latest_message = next_message
        next_message = queue.tryGet()
    return latest_message


def create_live_stream_state(
    *,
    device_id: str,
    camera_role: str,
    args: argparse.Namespace,
    stack: ExitStack,
    shelf_config: ShelfWatchingConfig | None = None,
) -> LiveSyncedStreamState:
    device = resolve_live_device(device_id)
    calibration = device.readCalibration()
    intrinsics = intrinsics_from_matrix(
        calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            (args.processing_width, args.processing_height),
        )
    )

    pipeline = stack.enter_context(dai.Pipeline(device))
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_output = cam_rgb.requestOutput(
        size=(args.frame_width, args.frame_height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=args.fps,
    )
    processing_rgb_output = cam_rgb.requestOutput(
        size=(args.processing_width, args.processing_height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=args.fps,
    )

    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    median_filter = {
        "off": dai.MedianFilter.MEDIAN_OFF,
        "3x3": dai.MedianFilter.KERNEL_3x3,
        "5x5": dai.MedianFilter.KERNEL_5x5,
        "7x7": dai.MedianFilter.KERNEL_7x7,
    }[args.depth_median_filter]
    stereo.initialConfig.setMedianFilter(median_filter)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(args.processing_width, args.processing_height)
    mono_left.requestFullResolutionOutput(fps=args.fps).link(stereo.left)
    mono_right.requestFullResolutionOutput(fps=args.fps).link(stereo.right)

    processing_buffer_frames = max(8, int(math.ceil(args.fps * args.processing_buffer_seconds)))
    rgb_queue = rgb_output.createOutputQueue(maxSize=8, blocking=False)
    processing_rgb_queue = processing_rgb_output.createOutputQueue(
        maxSize=8,
        blocking=False,
    )
    depth_queue = stereo.depth.createOutputQueue(
        maxSize=processing_buffer_frames,
        blocking=False,
    )
    tracker = build_person_tracker(args)
    state = LiveSyncedStreamState(
        device_id=device_id,
        camera_role=camera_role,
        device=device,
        pipeline=pipeline,
        rgb_queue=rgb_queue,
        processing_rgb_queue=processing_rgb_queue,
        depth_queue=depth_queue,
        intrinsics=intrinsics,
        tracker=tracker,
        recent_raw_rgb_messages=deque(maxlen=8),
        recent_processing_rgb_messages=deque(maxlen=8),
        recent_rgb_track_snapshots=deque(maxlen=processing_buffer_frames),
    )
    if shelf_config is not None and is_observer_enabled(camera_role):
        shelf_anchor_manager = load_saved_shelf_anchor_manager(
            device_id=device_id,
            config=shelf_config,
            calibration_root=args.shelf_calibrations_root,
        )
        if shelf_anchor_manager is not None:
            state.shelf_anchor_manager = shelf_anchor_manager
            print(
                f"Configured shelf watching device_id={device_id} "
                f"catalog_shelves={len(shelf_anchor_manager.shelves)} "
                f"anchors={len(shelf_anchor_manager.anchors)} mode=saved_anchors"
            )
        else:
            calibration_path = (
                args.shelf_calibrations_root
                / f"shelf_anchors_{device_id}.json"
            )
            print(
                f"Skipping shelf watching device_id={device_id} "
                f"reason=no_saved_anchors calibration={calibration_path}"
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
    print(
        f"Started live RGBD stream device_id={device_id} role={camera_role} "
        f"{format_usb_connection(device)}"
    )
    return state


def load_saved_shelf_anchor_manager(
    *,
    device_id: str,
    config: ShelfWatchingConfig,
    calibration_root: Path,
) -> ShelfAnchorManager | None:
    manager = ShelfAnchorManager(
        device_id=device_id,
        config=config,
        calibration_root=calibration_root,
        auto_save=False,
    )
    return manager if manager.anchors else None


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
        "processing_width": args.processing_width,
        "processing_height": args.processing_height,
        "fps": args.fps,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"shop_api_key", "operator_api_token"}
        },
    }
    (artifact_writer.output_dir / "live_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def operator_runtime_configuration(
    *,
    args: argparse.Namespace,
    camera_roles: list[str],
) -> dict[str, object]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=True,
            text=True,
            timeout=2.0,
        )
        git_commit = result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    return {
        "gitCommit": git_commit,
        "deviceIds": list(args.device_id),
        "cameraRoles": list(camera_roles),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"shop_api_key", "operator_api_token"}
        },
    }


def drain_messages_into_buffer(queue: dai.MessageQueue, messages: deque[Any]) -> int:
    drained = 0
    message = queue.tryGet()
    while message is not None:
        messages.append(message)
        drained += 1
        message = queue.tryGet()
    return drained


def pop_latest_matching_rgb_pair(
    raw_messages: deque[Any],
    processing_messages: deque[Any],
) -> tuple[Any, Any] | None:
    if not raw_messages or not processing_messages:
        return None

    processing_indices_by_sequence = {
        int(message.getSequenceNum()): index
        for index, message in enumerate(processing_messages)
    }
    matching_pairs = [
        (raw_index, processing_indices_by_sequence[int(raw_message.getSequenceNum())])
        for raw_index, raw_message in enumerate(raw_messages)
        if int(raw_message.getSequenceNum()) in processing_indices_by_sequence
    ]
    if not matching_pairs:
        return None

    raw_index, processing_index = max(
        matching_pairs,
        key=lambda pair: float(raw_messages[pair[0]].getTimestamp().total_seconds()),
    )
    for _index in range(raw_index):
        raw_messages.popleft()
    raw_message = raw_messages.popleft()
    for _index in range(processing_index):
        processing_messages.popleft()
    processing_message = processing_messages.popleft()
    return raw_message, processing_message


def pop_matching_rgb_track_snapshot(
    snapshots: deque[LiveRgbTrackSnapshot],
    *,
    depth_sequence_num: int,
    depth_host_synced_seconds: float,
    max_delta_ms: float,
) -> tuple[LiveRgbTrackSnapshot | None, float | None]:
    if not snapshots:
        return None, None

    exact_index = next(
        (
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot.sequence_num == depth_sequence_num
        ),
        None,
    )
    selected_index = exact_index
    if exact_index is not None:
        exact_delta_ms = (
            depth_host_synced_seconds - snapshots[exact_index].host_synced_seconds
        ) * 1000.0
        if abs(exact_delta_ms) > max_delta_ms:
            selected_index = None
    if selected_index is None:
        selected_index = min(
            range(len(snapshots)),
            key=lambda index: abs(
                snapshots[index].host_synced_seconds - depth_host_synced_seconds
            ),
        )

    selected_snapshot = snapshots[selected_index]
    delta_ms = (
        depth_host_synced_seconds - selected_snapshot.host_synced_seconds
    ) * 1000.0
    if abs(delta_ms) > max_delta_ms:
        return None, delta_ms

    for _index in range(selected_index):
        snapshots.popleft()
    return snapshots.popleft(), delta_ms


def drain_latest_depth_message_for_snapshots(
    queue: dai.MessageQueue,
    snapshots: deque[LiveRgbTrackSnapshot],
    *,
    max_delta_ms: float,
) -> tuple[Any | None, Any | None]:
    latest_message = None
    latest_matching_message = None
    latest_matching_key: tuple[float, float] | None = None
    message = queue.tryGet()
    while message is not None:
        latest_message = message
        if snapshots:
            depth_sequence = int(message.getSequenceNum())
            depth_host_seconds = float(message.getTimestamp().total_seconds())
            exact_snapshot = next(
                (
                    snapshot
                    for snapshot in snapshots
                    if snapshot.sequence_num == depth_sequence
                ),
                None,
            )
            if exact_snapshot is not None:
                exact_delta_ms = (
                    depth_host_seconds - exact_snapshot.host_synced_seconds
                ) * 1000.0
                if abs(exact_delta_ms) > max_delta_ms:
                    exact_snapshot = None
            matched_snapshot = exact_snapshot or min(
                snapshots,
                key=lambda snapshot: abs(
                    snapshot.host_synced_seconds - depth_host_seconds
                ),
            )
            delta_ms = (
                depth_host_seconds - matched_snapshot.host_synced_seconds
            ) * 1000.0
            if abs(delta_ms) <= max_delta_ms:
                candidate_key = (
                    matched_snapshot.host_synced_seconds,
                    -abs(delta_ms),
                )
                if latest_matching_key is None or candidate_key > latest_matching_key:
                    latest_matching_message = message
                    latest_matching_key = candidate_key
        message = queue.tryGet()
    return latest_matching_message, latest_message


def update_cached_depth_overlay(
    *,
    state: LiveSyncedStreamState,
    depth_frame_mm: np.ndarray,
    hide_depth_window: bool,
    performance: LivePerformanceLogger,
) -> None:
    if hide_depth_window:
        state.cached_depth_overlay = None
        return

    depth_colorize_started = performance.start()
    state.cached_depth_overlay = colorize_depth(depth_frame_mm)
    performance.record_duration("depth_colorize", depth_colorize_started)


def rgb_depth_delta_ms(
    depth_packet: LiveDepthPacket,
    rgb_host_synced_seconds: float,
) -> float:
    return (depth_packet.host_synced_seconds - rgb_host_synced_seconds) * 1000.0


def rgb_depth_pair_is_synchronized(
    depth_packet: LiveDepthPacket,
    rgb_host_synced_seconds: float,
    *,
    max_delta_ms: float,
) -> bool:
    return abs(rgb_depth_delta_ms(depth_packet, rgb_host_synced_seconds)) <= max_delta_ms


def process_latest_rgb_pair(
    *,
    state: LiveSyncedStreamState,
    detector: PersonDetector,
    face_matcher: FaceRecognizer | None,
    body_evidence_extractor: BodyEvidenceExtractor,
    args: argparse.Namespace,
    performance: LivePerformanceLogger,
) -> bool:
    rgb_poll_started = performance.start()
    raw_drained = drain_messages_into_buffer(
        state.rgb_queue,
        state.recent_raw_rgb_messages,
    )
    processing_drained = drain_messages_into_buffer(
        state.processing_rgb_queue,
        state.recent_processing_rgb_messages,
    )
    performance.record_metric("raw_rgb_queue_drained", float(raw_drained))
    performance.record_metric(
        "processing_rgb_queue_drained",
        float(processing_drained),
    )
    matched_pair = pop_latest_matching_rgb_pair(
        state.recent_raw_rgb_messages,
        state.recent_processing_rgb_messages,
    )
    performance.record_duration("rgb_poll", rgb_poll_started)
    if matched_pair is None:
        return False

    raw_rgb_msg, processing_rgb_msg = matched_pair
    raw_rgb_host_synced_seconds = float(raw_rgb_msg.getTimestamp().total_seconds())
    processing_rgb_host_synced_seconds = float(
        processing_rgb_msg.getTimestamp().total_seconds()
    )
    captured_age_reference = time.monotonic()
    performance.record_metric(
        "raw_rgb_capture_age_ms",
        max(0.0, captured_age_reference - raw_rgb_host_synced_seconds) * 1000.0,
    )
    performance.record_metric(
        "processing_rgb_capture_age_ms",
        max(0.0, captured_age_reference - processing_rgb_host_synced_seconds) * 1000.0,
    )
    performance.record_metric(
        "rgb_pair_delta_ms",
        abs(raw_rgb_host_synced_seconds - processing_rgb_host_synced_seconds) * 1000.0,
    )
    rgb_sequence = int(processing_rgb_msg.getSequenceNum())
    if state.last_rgb_evidence_sequence == rgb_sequence:
        return False

    rgb_decode_started = performance.start()
    raw_rgb_frame = raw_rgb_msg.getCvFrame()
    processing_rgb_frame = processing_rgb_msg.getCvFrame()
    performance.record_duration("rgb_decode", rgb_decode_started)
    state.cached_rgb_raw = raw_rgb_frame
    state.last_raw_rgb_sequence = int(raw_rgb_msg.getSequenceNum())
    if state.cached_rgb_overlay is None:
        state.cached_rgb_overlay = raw_rgb_frame

    yolo_started = performance.start()
    detections = detector.detect(processing_rgb_frame)
    performance.record_duration("yolo", yolo_started)
    tracking_started = performance.start()
    current_processing_tracks = state.tracker.update(detections)
    processing_height, processing_width = processing_rgb_frame.shape[:2]
    processing_tracks = scale_tracks(
        current_processing_tracks,
        source_width=processing_width,
        source_height=processing_height,
        target_width=processing_width,
        target_height=processing_height,
    )
    display_tracks = scale_tracks(
        processing_tracks,
        source_width=processing_width,
        source_height=processing_height,
        target_width=raw_rgb_frame.shape[1],
        target_height=raw_rgb_frame.shape[0],
    )
    performance.record_duration("tracking", tracking_started)
    performance.record_processed_frame(
        detection_count=len(detections),
        track_count=len(processing_tracks),
    )

    face_started = performance.start()
    recognized_faces: list[RecognizedFace] = []
    if face_matcher is not None:
        eligible_processing_tracks = face_recognition_eligible_tracks(
            processing_tracks,
            min_width_px=args.face_min_track_width_px,
            min_height_px=args.face_min_track_height_px,
        )
        eligible_track_ids = {
            track.track_id for track in eligible_processing_tracks
        }
        eligible_display_tracks = [
            track
            for track in display_tracks
            if track.track_id in eligible_track_ids
        ]
        recognized_faces = face_matcher.recognize_crops(
            raw_rgb_frame,
            tracks=eligible_display_tracks,
        )
    performance.record_duration("face", face_started)

    body_started = performance.start()
    processing_body_evidence_by_track = body_evidence_extractor.extract(
        processing_rgb_frame,
        tracks=processing_tracks,
    )
    body_evidence_by_track = scale_body_evidence_heights(
        processing_body_evidence_by_track,
        scale_y=raw_rgb_frame.shape[0] / processing_height,
    )
    performance.record_duration("body", body_started)

    rgb_host_synced_seconds = processing_rgb_host_synced_seconds
    state.recent_rgb_track_snapshots.append(
        LiveRgbTrackSnapshot(
            sequence_num=rgb_sequence,
            host_synced_seconds=rgb_host_synced_seconds,
            processing_frame=processing_rgb_frame,
            processing_tracks=tuple(processing_tracks),
            display_tracks=tuple(display_tracks),
            recognized_faces=tuple(recognized_faces),
            body_evidence_by_track=body_evidence_by_track,
        )
    )
    performance.record_rgb_frame()
    state.last_raw_rgb_host_synced_seconds = raw_rgb_host_synced_seconds
    state.last_rgb_host_synced_seconds = rgb_host_synced_seconds
    state.last_rgb_evidence_sequence = rgb_sequence
    return True


def process_latest_depth_frame(
    *,
    camera_index: int,
    state: LiveSyncedStreamState,
    visit_registry: VisitRegistry,
    artifact_writer: ReplayArtifactWriter,
    shop_api_client: ShopApiClient,
    shop_state_store: ShopStateStore,
    customer_ids_by_visit: dict[int, str],
    shelf_coordinator: ShelfProximityCoordinator | None,
    stream_server: MjpegStreamServer | None,
    args: argparse.Namespace,
    performance: LivePerformanceLogger,
) -> bool:
    depth_msg, latest_depth_msg = drain_latest_depth_message_for_snapshots(
        state.depth_queue,
        state.recent_rgb_track_snapshots,
        max_delta_ms=args.max_rgb_depth_delta_ms,
    )
    if depth_msg is None:
        if latest_depth_msg is not None:
            latest_depth_seconds = float(
                latest_depth_msg.getTimestamp().total_seconds()
            )
            if (
                state.last_depth_sync_warning_seconds is None
                or latest_depth_seconds - state.last_depth_sync_warning_seconds >= 5.0
            ):
                print(
                    f"LIVE_RGB_DEPTH_MATCH_PENDING device_id={state.device_id} "
                    f"latest_depth_sequence_num={int(latest_depth_msg.getSequenceNum())} "
                    f"buffered_rgb_snapshots={len(state.recent_rgb_track_snapshots)}"
                )
                state.last_depth_sync_warning_seconds = latest_depth_seconds
        return False

    depth_sequence = int(depth_msg.getSequenceNum())
    depth_host_synced_seconds = float(depth_msg.getTimestamp().total_seconds())
    rgb_snapshot, matched_delta_ms = pop_matching_rgb_track_snapshot(
        state.recent_rgb_track_snapshots,
        depth_sequence_num=depth_sequence,
        depth_host_synced_seconds=depth_host_synced_seconds,
        max_delta_ms=args.max_rgb_depth_delta_ms,
    )
    if rgb_snapshot is None:
        if (
            state.last_depth_sync_warning_seconds is None
            or depth_host_synced_seconds - state.last_depth_sync_warning_seconds >= 5.0
        ):
            delta_text = "none" if matched_delta_ms is None else f"{matched_delta_ms:.1f}"
            print(
                f"LIVE_RGB_DEPTH_SYNC_REJECTED device_id={state.device_id} "
                f"depth_sequence_num={depth_sequence} delta_ms={delta_text} "
                f"max_delta_ms={args.max_rgb_depth_delta_ms:.1f} "
                f"buffered_rgb_snapshots={len(state.recent_rgb_track_snapshots)}"
            )
            state.last_depth_sync_warning_seconds = depth_host_synced_seconds
        return False

    rgb_sequence = rgb_snapshot.sequence_num
    if state.last_processed_rgb_sequence == rgb_sequence:
        return False

    depth_drain_started = performance.start()
    best_depth = LiveDepthPacket(
        sequence_num=depth_sequence,
        host_synced_seconds=depth_host_synced_seconds,
        device_monotonic_seconds=float(depth_msg.getTimestampDevice().total_seconds()),
        frame_mm=depth_msg.getFrame().copy(),
    )
    performance.record_duration("depth_drain", depth_drain_started)
    state.last_processed_rgb_sequence = rgb_sequence

    if not rgb_depth_pair_is_synchronized(
        best_depth,
        rgb_snapshot.host_synced_seconds,
        max_delta_ms=args.max_rgb_depth_delta_ms,
    ):
        return False

    processed_frame = build_processed_live_rgb_frame(
        camera_index=camera_index,
        state=state,
        rgb_snapshot=rgb_snapshot,
        depth_frame_mm=best_depth.frame_mm,
        depth_packet=best_depth,
        visit_registry=visit_registry,
        artifact_writer=artifact_writer,
        shop_api_client=shop_api_client,
        shop_state_store=shop_state_store,
        customer_ids_by_visit=customer_ids_by_visit,
        shelf_coordinator=shelf_coordinator,
        stream_server=stream_server,
        args=args,
        performance=performance,
    )
    state.cached_rgb_overlay = processed_frame.overlay
    state.cached_observer_snapshot = processed_frame.observer_snapshot
    state.cached_shelf_snapshot = processed_frame.shelf_snapshot
    state.pending_shelf_events.extend(processed_frame.shelf_events)
    update_cached_depth_overlay(
        state=state,
        depth_frame_mm=best_depth.frame_mm,
        hide_depth_window=args.hide_depth_window,
        performance=performance,
    )
    return True


def build_shelf_camera_observations(
    *,
    camera_index: int,
    state: LiveSyncedStreamState,
    processing_tracks: list[Track],
    shelf_person_depth_samples: dict[int, DepthSample],
    visit_assignments: dict[int, VisitAssignment],
    customer_ids_by_visit: dict[int, str],
    host_synced_seconds: float,
    observed_at_unix_milliseconds: int,
    rgb_sequence_number: int,
    depth_sequence_number: int,
) -> tuple[ShelfCameraObservation, ...]:
    manager = state.shelf_anchor_manager
    if manager is None:
        return ()
    closest_observations: dict[tuple[int, int], ShelfCameraObservation] = {}
    for shelf in manager.shelves:
        anchors = manager.anchors_for_shelf(shelf.shelf_id)
        if not anchors:
            continue
        for track in processing_tracks:
            if track.status not in {"NEW", "TRACKED"}:
                continue
            sample = shelf_person_depth_samples.get(track.track_id)
            if sample is None:
                continue
            assignment = visit_assignments.get(track.track_id)
            visit_id = None if assignment is None else assignment.visit_id
            for anchor in anchors:
                observation = ShelfCameraObservation(
                    shelf_id=shelf.shelf_id,
                    shelf_label=shelf.label,
                    marker_id=anchor.marker_id,
                    camera_index=camera_index,
                    device_id=state.device_id,
                    track_id=track.track_id,
                    visit_id=visit_id,
                    visit_origin=None if assignment is None else assignment.origin,
                    customer_id=(
                        None
                        if visit_id is None
                        else customer_ids_by_visit.get(visit_id)
                    ),
                    distance_mm=person_to_shelf_distance_mm(
                        sample.point_3d_mm,
                        anchor.point_3d_mm,
                    ),
                    person_point_3d_mm=sample.point_3d_mm,
                    anchor=anchor,
                    host_synced_seconds=host_synced_seconds,
                    observed_at_unix_milliseconds=observed_at_unix_milliseconds,
                    rgb_sequence_number=rgb_sequence_number,
                    depth_sequence_number=depth_sequence_number,
                )
                key = (shelf.shelf_id, track.track_id)
                current = closest_observations.get(key)
                if current is None or observation.distance_mm < current.distance_mm:
                    closest_observations[key] = observation
    return tuple(closest_observations.values())


def persist_shelf_events(
    events: tuple[ShelfProximityEvent, ...],
    *,
    shop_state_store: ShopStateStore,
    artifact_writer: ReplayArtifactWriter,
) -> tuple[ShelfProximityEvent, ...]:
    persisted: list[ShelfProximityEvent] = []
    for event in events:
        persisted_event = shop_state_store.record_shelf_event(event)
        payload = shelf_event_payload(persisted_event)
        artifact_writer.write_shelf_event(payload)
        prefix = (
            "SHELF_APPROACH_EVENT"
            if event.event_type == "shelf_approach"
            else "SHELF_DEPARTURE_EVENT"
        )
        print(
            f"{prefix} event_id={persisted_event.event_id} "
            f"shelf_id={event.shelf_id} visit_id={event.visit_id} "
            f"customer_id={event.customer_id} device_id={event.device_id} "
            f"track_id={event.track_id} distance_mm={event.distance_mm:.0f} "
            f"session_id={event.proximity_session_id} reason={event.reason}"
        )
        persisted.append(persisted_event)
    return tuple(persisted)


def build_processed_live_rgb_frame(
    *,
    camera_index: int,
    state: LiveSyncedStreamState,
    rgb_snapshot: LiveRgbTrackSnapshot,
    depth_frame_mm: np.ndarray,
    depth_packet: LiveDepthPacket,
    visit_registry: VisitRegistry,
    artifact_writer: ReplayArtifactWriter,
    shop_api_client: ShopApiClient,
    shop_state_store: ShopStateStore,
    customer_ids_by_visit: dict[int, str],
    shelf_coordinator: ShelfProximityCoordinator | None,
    stream_server: MjpegStreamServer | None,
    args: argparse.Namespace,
    performance: LivePerformanceLogger,
) -> ProcessedLiveFrame:
    processing_rgb_frame = rgb_snapshot.processing_frame
    processing_tracks = list(rgb_snapshot.processing_tracks)
    tracks = list(rgb_snapshot.display_tracks)
    recognized_faces = list(rgb_snapshot.recognized_faces)
    body_evidence_by_track = rgb_snapshot.body_evidence_by_track
    rgb_host_synced_seconds = rgb_snapshot.host_synced_seconds
    rgb_sequence_num = rgb_snapshot.sequence_num
    processing_height, processing_width = processing_rgb_frame.shape[:2]
    display_width = args.frame_width
    display_height = args.frame_height
    processing_coordinate_scale = min(
        processing_width / display_width,
        processing_height / display_height,
    )
    entrance_enabled = is_entrance_enabled(state.camera_role)

    depth_logic_started = performance.start()
    if args.depth_trigger_mode == "plane" and entrance_enabled:
        depth_result = process_depth_plane_logic(
            tracks=processing_tracks,
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
            track_split_recovery_max_centroid_distance_px=(
                args.plane_track_split_recovery_max_centroid_distance_px
                * processing_coordinate_scale
            ),
        )
    else:
        depth_result = process_depth_entrance_logic(
            tracks=processing_tracks,
            depth_frame_mm=depth_frame_mm,
            intrinsics=state.intrinsics,
            states=state.depth_states,
            depth_threshold_mm=float(args.depth_threshold_mm),
            depth_hysteresis_mm=float(args.depth_hysteresis_mm),
            min_valid_pixels=args.depth_min_valid_pixels,
            roi_width_fraction=args.depth_roi_width_fraction,
            roi_height_fraction=args.depth_roi_height_fraction,
        )
    performance.record_duration("depth_logic", depth_logic_started)

    frame_unix_milliseconds = time.time_ns() // 1_000_000
    shelf_person_depth_samples: dict[int, DepthSample] = {}
    if state.shelf_anchor_manager is not None:
        shelf_depth_started = performance.start()
        for track in processing_tracks:
            if track.status not in {"NEW", "TRACKED"}:
                continue
            sample = sample_shelf_person_depth(
                depth_frame_mm,
                track,
                intrinsics=state.intrinsics,
                config=state.shelf_anchor_manager.config.person_depth,
            )
            if sample is not None:
                shelf_person_depth_samples[track.track_id] = sample
        performance.record_duration("shelf_depth", shelf_depth_started)

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
            tracks=processing_tracks,
            depth_samples=depth_samples,
            signed_distances_mm=signed_distances_mm,
            depth_states=state.depth_states,
            entered_track_ids=set(entered_track_ids),
            exited_track_ids=set(exited_track_ids),
        )

    registry_started = performance.start()
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
    visit_decisions: dict[int, VisitRegistryDecision] = {}
    shelf_events_this_frame: list[ShelfProximityEvent] = []
    observer_enabled = is_observer_enabled(state.camera_role)
    for track_id, track_evidence in track_visit_evidence_by_id.items():
        decision = visit_registry.resolve_existing_track(track_evidence)
        resolution = "existing_track"
        if decision is None and observer_enabled:
            decision = visit_registry.resolve_observer_track(track_evidence)
            resolution = "observer_track"
        if decision is not None:
            visit_assignments[track_id] = decision.assignment
            visit_decisions[track_id] = decision
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
            visit_decisions[track_id] = decision
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
        if stream_server is not None:
            stream_server.publish_entrance_event(
                event_type="entry_accepted",
                camera_index=camera_index,
                payload=event_payload,
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
                if stream_server is not None:
                    stream_server.publish_customer_binding(
                        visit_id=visit_id,
                        customer_id=shopping_customer_id,
                        host_synced_seconds=rgb_host_synced_seconds,
                        camera_index=camera_index,
                        device_id=state.device_id,
                        track_id=track_id,
                    )
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
        if stream_server is not None:
            stream_server.publish_entrance_event(
                event_type="leave_accepted",
                camera_index=camera_index,
                payload=event_payload,
            )
        if shelf_coordinator is not None:
            closed_shelf_events = shelf_coordinator.close_visit(
                visit_id,
                host_synced_seconds=rgb_host_synced_seconds,
                now_unix_milliseconds=frame_unix_milliseconds,
            )
            shelf_events_this_frame.extend(
                persist_shelf_events(
                    closed_shelf_events,
                    shop_state_store=shop_state_store,
                    artifact_writer=artifact_writer,
                )
            )
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

    shelf_snapshot = None
    if state.shelf_anchor_manager is not None:
        shelf_coordinator_started = performance.start()
        shelf_observations = build_shelf_camera_observations(
            camera_index=camera_index,
            state=state,
            processing_tracks=processing_tracks,
            shelf_person_depth_samples=shelf_person_depth_samples,
            visit_assignments=visit_assignments,
            customer_ids_by_visit=customer_ids_by_visit,
            host_synced_seconds=rgb_host_synced_seconds,
            observed_at_unix_milliseconds=frame_unix_milliseconds,
            rgb_sequence_number=rgb_sequence_num,
            depth_sequence_number=depth_packet.sequence_num,
        )
        if args.log_shelf_distance:
            closest_by_shelf: dict[int, ShelfCameraObservation] = {}
            for observation in shelf_observations:
                current = closest_by_shelf.get(observation.shelf_id)
                if current is None or observation.distance_mm < current.distance_mm:
                    closest_by_shelf[observation.shelf_id] = observation
            for observation in closest_by_shelf.values():
                print(
                    f"SHELF_DISTANCE_TRACE shelf_id={observation.shelf_id} "
                    f"visit_id={observation.visit_id} device_id={observation.device_id} "
                    f"track_id={observation.track_id} "
                    f"distance_mm={observation.distance_mm:.0f}"
                )
        if shelf_coordinator is not None:
            coordinator_events = shelf_coordinator.update_camera(
                camera_index=camera_index,
                observations=shelf_observations,
                host_synced_seconds=rgb_host_synced_seconds,
                now_unix_milliseconds=frame_unix_milliseconds,
            )
            shelf_events_this_frame.extend(
                persist_shelf_events(
                    coordinator_events,
                    shop_state_store=shop_state_store,
                    artifact_writer=artifact_writer,
                )
            )
        shelf_snapshot = build_shelf_camera_snapshot(
            camera_index=camera_index,
            device_id=state.device_id,
            camera_role=state.camera_role,
            rgb_sequence_number=rgb_sequence_num,
            depth_sequence_number=depth_packet.sequence_num,
            host_synced_seconds=rgb_host_synced_seconds,
            shelves=state.shelf_anchor_manager.anchored_shelves,
            anchors_by_shelf=state.shelf_anchor_manager.anchors_by_shelf,
            observations=shelf_observations,
            states_by_shelf=(
                {}
                if shelf_coordinator is None
                else {
                    status.shelf_id: status.state
                    for status in shelf_coordinator.statuses(
                        now_unix_milliseconds=frame_unix_milliseconds
                    )
                }
            ),
        )
        performance.record_duration("shelf_coordinator", shelf_coordinator_started)

    observer_snapshot = None
    display_depth_samples = scale_depth_samples(
        depth_samples,
        source_width=processing_width,
        source_height=processing_height,
        target_width=display_width,
        target_height=display_height,
    )
    if observer_enabled:
        observer_snapshot = build_observer_camera_snapshot(
            camera_index=camera_index,
            device_id=state.device_id,
            camera_role=state.camera_role,
            rgb_frame=processing_rgb_frame,
            rgb_sequence_number=rgb_sequence_num,
            host_synced_seconds=rgb_host_synced_seconds,
            tracks=tracks,
            track_visit_evidence_by_id=track_visit_evidence_by_id,
            visit_assignments=visit_assignments,
            depth_samples=display_depth_samples,
            customer_ids_by_visit=customer_ids_by_visit,
            visit_decisions=visit_decisions,
            frame_width=display_width,
            frame_height=display_height,
            provisional_track_ids={
                track_id
                for track_id in track_visit_evidence_by_id
                if visit_registry.is_observer_track_provisional(
                    device_id=state.device_id,
                    track_id=track_id,
                )
            },
        )
    performance.record_duration("registry_io", registry_started)

    overlay_started = performance.start()
    processing_overlay = processing_rgb_frame.copy()
    draw_tracks(processing_overlay, processing_tracks)
    draw_visit_labels(
        processing_overlay,
        processing_tracks,
        visit_assignments,
        show_face_evidence=False,
    )
    draw_depth_samples(
        processing_overlay,
        tracks=processing_tracks,
        depth_samples=depth_samples,
        depth_threshold_mm=float(args.depth_threshold_mm),
        signed_distances_mm=signed_distances_mm,
        plane_mode=args.depth_trigger_mode == "plane",
    )
    if recognized_faces:
        processing_faces = scale_recognized_faces(
            recognized_faces,
            source_width=display_width,
            source_height=display_height,
            target_width=processing_width,
            target_height=processing_height,
        )
        draw_recognized_faces(processing_overlay, processing_faces, show_labels=False)
    if args.stream_annotated or args.show_annotated_preview:
        overlay = cv2.resize(
            processing_overlay,
            (display_width, display_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        overlay = processing_overlay
    performance.record_duration("overlay", overlay_started)
    return ProcessedLiveFrame(
        overlay=overlay,
        observer_snapshot=observer_snapshot,
        shelf_snapshot=shelf_snapshot,
        shelf_events=tuple(shelf_events_this_frame),
    )


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


def select_preview_frame(
    state: LiveSyncedStreamState,
    *,
    show_annotated_preview: bool,
) -> np.ndarray | None:
    if show_annotated_preview and state.cached_rgb_overlay is not None:
        return state.cached_rgb_overlay
    if state.cached_rgb_raw is not None:
        return state.cached_rgb_raw
    return state.cached_rgb_overlay


def preview_frame_or_placeholder(
    state: LiveSyncedStreamState,
    *,
    label: str,
    width: int,
    height: int,
    show_annotated_preview: bool,
) -> np.ndarray:
    selected = select_preview_frame(
        state,
        show_annotated_preview=show_annotated_preview,
    )
    if selected is not None:
        return selected
    return placeholder_frame(label, width=width, height=height)


def restore_shelf_proximity_sessions(
    coordinator: ShelfProximityCoordinator,
    payloads: list[dict[str, Any]],
) -> None:
    restored = 0
    for payload in payloads:
        try:
            camera = payload["camera"]
            anchor_payload = payload["anchor"]
            anchor_point = anchor_payload["point3dMm"]
            person_point = payload["personPoint3dMm"]
            anchor = ShelfAnchor(
                shelf_id=int(payload["shelfId"]),
                marker_id=int(payload["markerId"]),
                device_id=str(camera["deviceId"]),
                point_3d_mm=(
                    float(anchor_point["x"]),
                    float(anchor_point["y"]),
                    float(anchor_point["z"]),
                ),
                sample_count=int(anchor_payload.get("sampleCount", 0)),
                rms_spread_mm=float(anchor_payload.get("rmsSpreadMm", 0.0)),
                updated_at_unix_milliseconds=int(
                    anchor_payload.get("updatedAtUnixMilliseconds", 0)
                ),
                source=str(anchor_payload.get("source", "persisted")),
            )
            observation = ShelfCameraObservation(
                shelf_id=int(payload["shelfId"]),
                shelf_label=str(payload["shelfLabel"]),
                marker_id=int(payload["markerId"]),
                camera_index=int(camera["id"]),
                device_id=str(camera["deviceId"]),
                track_id=int(camera["trackId"]),
                visit_id=int(payload["visitId"]),
                visit_origin=payload.get("visitOrigin"),
                customer_id=payload.get("customerId"),
                distance_mm=float(payload["distanceMm"]),
                person_point_3d_mm=(
                    float(person_point["x"]),
                    float(person_point["y"]),
                    float(person_point["z"]),
                ),
                anchor=anchor,
                host_synced_seconds=float(payload["hostSyncedSeconds"]),
                observed_at_unix_milliseconds=int(
                    payload["occurredAtUnixMilliseconds"]
                ),
                rgb_sequence_number=int(payload["rgbSequenceNumber"]),
                depth_sequence_number=int(payload["depthSequenceNumber"]),
            )
            coordinator.restore_near_session(
                shelf_id=observation.shelf_id,
                visit_id=int(payload["visitId"]),
                proximity_session_id=str(payload["proximitySessionId"]),
                observation=observation,
                minimum_distance_mm=(
                    None
                    if payload.get("minimumDistanceMm") is None
                    else float(payload["minimumDistanceMm"])
                ),
            )
            restored += 1
        except (KeyError, TypeError, ValueError) as exc:
            print(f"SHELF_SESSION_RESTORE_SKIPPED error={exc}")
    if restored:
        print(f"Restored {restored} active shelf proximity sessions.")


def main() -> None:
    args = build_argparser().parse_args()
    validate_operator_console_args(args)
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise ValueError("--frame-width and --frame-height must be greater than zero.")
    if args.processing_width <= 0 or args.processing_height <= 0:
        raise ValueError("--processing-width and --processing-height must be greater than zero.")
    if args.max_rgb_depth_delta_ms < 0.0:
        raise ValueError("--max-rgb-depth-delta-ms must be zero or greater.")
    if args.processing_buffer_seconds <= 0.0:
        raise ValueError("--processing-buffer-seconds must be greater than zero.")
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
    if args.performance_log_interval_seconds <= 0:
        raise ValueError("--performance-log-interval-seconds must be greater than zero.")

    stop_requested = threading.Event()
    performance = LivePerformanceLogger(
        enabled=args.log_performance,
        interval_seconds=args.performance_log_interval_seconds,
    )

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    camera_roles = resolve_camera_roles(args)
    shelf_config = None
    shelf_coordinator = None
    if args.enable_shelf_watching:
        shelf_config = load_shelf_config(args.shelf_config)
        validate_shelf_config_for_live_cameras(
            shelf_config,
            camera_device_ids=args.device_id,
            observer_capable_device_ids={
                device_id
                for device_id, camera_role in zip(args.device_id, camera_roles)
                if is_observer_enabled(camera_role)
            },
        )
        shelf_coordinator = ShelfProximityCoordinator(shelf_config)
        print(
            f"Loaded shelf config {args.shelf_config} "
            f"shelves={len(shelf_config.shelves)}"
        )
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
        observer_single_active_fallback_threshold=args.observer_single_active_fallback_threshold,
        observer_provisional_seconds=args.observer_provisional_seconds,
        log_decisions=args.log_visit_decisions,
    )
    shop_state_store = ShopStateStore(args.state_db)
    customer_ids_by_visit = shop_state_store.load_shop_customer_bindings()
    visit_registry.next_visit_id = max(visit_registry.next_visit_id, shop_state_store.next_visit_id())
    print(
        f"Using shop state DB {shop_state_store.db_path} "
        f"next_visit_id={visit_registry.next_visit_id}"
    )
    if shelf_coordinator is not None:
        restore_shelf_proximity_sessions(
            shelf_coordinator,
            shop_state_store.load_active_shelf_session_payloads(),
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
                operator_state_db=(
                    args.state_db if args.enable_operator_console else None
                ),
                operator_runs_root=args.operator_runs_root,
                operator_api_token=args.operator_api_token,
                operator_runtime_configuration=operator_runtime_configuration(
                    args=args,
                    camera_roles=camera_roles,
                ),
            )
            stream_server.start()
            stream_server.publish_shelf_event_payloads(
                shop_state_store.load_recent_shelf_event_payloads(),
                emit_operator_events=False,
            )
            if args.enable_operator_console:
                print(
                    f"Operator console available at "
                    f"http://{args.stream_host}:{args.stream_port}/operator/ "
                    "(bearer token required)"
                )
            if shelf_coordinator is not None:
                stream_server.publish_shelf_statuses(
                    shelf_coordinator.statuses(
                        now_unix_milliseconds=time.time_ns() // 1_000_000
                    )
                )

        with ExitStack() as stack:
            states = [
                create_live_stream_state(
                    device_id=device_id,
                    camera_role=camera_role,
                    args=args,
                    stack=stack,
                    shelf_config=shelf_config,
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
                cycle_started = performance.start()
                for camera_index, state in enumerate(states):
                    camera_iteration_started = performance.start()
                    performance.record_camera_poll()
                    if state.device.isClosed() or not state.pipeline.isRunning():
                        raise RuntimeError(f"Live pipeline stopped for device {state.device_id}.")
                    previous_raw_rgb_sequence = state.last_raw_rgb_sequence
                    previous_processed_rgb_sequence = state.last_processed_rgb_sequence
                    process_latest_rgb_pair(
                        state=state,
                        detector=detector,
                        face_matcher=face_matcher,
                        body_evidence_extractor=body_evidence_extractor,
                        args=args,
                        performance=performance,
                    )
                    process_latest_depth_frame(
                        camera_index=camera_index,
                        state=state,
                        visit_registry=visit_registry,
                        artifact_writer=artifact_writer,
                        shop_api_client=shop_api_client,
                        shop_state_store=shop_state_store,
                        customer_ids_by_visit=customer_ids_by_visit,
                        shelf_coordinator=shelf_coordinator,
                        stream_server=stream_server,
                        args=args,
                        performance=performance,
                    )
                    raw_rgb_changed = state.last_raw_rgb_sequence != previous_raw_rgb_sequence
                    processed_rgb_changed = (
                        state.last_processed_rgb_sequence != previous_processed_rgb_sequence
                    )
                    if stream_server is not None and (
                        (args.stream_annotated and processed_rgb_changed)
                        or (not args.stream_annotated and raw_rgb_changed)
                        or processed_rgb_changed
                    ):
                        stream_started = performance.start()
                        stream_frame = (
                            state.cached_rgb_overlay
                            if args.stream_annotated
                            else state.cached_rgb_raw
                        )
                        if stream_frame is not None and (
                            (args.stream_annotated and processed_rgb_changed)
                            or (not args.stream_annotated and raw_rgb_changed)
                        ):
                            stream_server.publish(
                                camera_index,
                                stream_frame,
                                rgb_sequence_number=(
                                    state.last_processed_rgb_sequence
                                    if args.stream_annotated
                                    else state.last_raw_rgb_sequence
                                ),
                            )
                            performance.record_stream_frame()
                        if processed_rgb_changed and state.cached_observer_snapshot is not None:
                            stream_server.publish_observer_snapshot(
                                camera_index,
                                state.cached_observer_snapshot,
                            )
                        if processed_rgb_changed and state.cached_shelf_snapshot is not None:
                            stream_server.publish_shelf_snapshot(
                                camera_index,
                                state.cached_shelf_snapshot,
                            )
                        if state.pending_shelf_events:
                            stream_server.publish_shelf_event_payloads(
                                [
                                    shelf_event_payload(event)
                                    for event in state.pending_shelf_events
                                ]
                            )
                            state.pending_shelf_events.clear()
                        if shelf_coordinator is not None and processed_rgb_changed:
                            stream_server.publish_shelf_statuses(
                                shelf_coordinator.statuses(
                                    now_unix_milliseconds=time.time_ns() // 1_000_000
                                )
                            )
                        performance.record_duration("stream_publish", stream_started)
                    elif state.pending_shelf_events:
                        state.pending_shelf_events.clear()
                    performance.record_duration("camera_iteration", camera_iteration_started)

                if not args.headless:
                    gui_started = performance.start()
                    preview_time = time.monotonic()
                    for state in states:
                        if state.last_raw_rgb_host_synced_seconds is not None:
                            performance.record_metric(
                                "preview_age_ms",
                                max(
                                    0.0,
                                    preview_time - state.last_raw_rgb_host_synced_seconds,
                                )
                                * 1000.0,
                            )
                    rgb_frames = [
                        preview_frame_or_placeholder(
                            state,
                            label=f"Camera {index + 1}: waiting",
                            width=args.frame_width,
                            height=args.frame_height,
                            show_annotated_preview=args.show_annotated_preview,
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
                                width=args.processing_width,
                                height=args.processing_height,
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
                    performance.record_duration("gui", gui_started)
                    if key == ord("q"):
                        break
                time.sleep(0.001)
                performance.complete_cycle(cycle_started)
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
