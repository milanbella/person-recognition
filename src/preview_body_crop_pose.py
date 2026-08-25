from __future__ import annotations

import argparse
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import depthai as dai
import numpy as np

from pipeline.camera import (
    configure_live_device,
    device_identifier,
    list_available_devices,
    print_available_devices,
    print_connected_device,
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
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
)
from pipeline.detection import YoloOnnxPersonDetector
from pipeline.pose import (
    DEFAULT_POSE_KEYPOINT_THRESHOLD,
    DEFAULT_POSE_MODEL,
    DEFAULT_POSE_SCORE_THRESHOLD,
    PoseObservation,
    YoloOnnxPoseEstimator,
    crop_person_for_pose,
    draw_pose_observation,
    map_pose_detection_to_source,
)
from pipeline.tracking import build_person_tracker, draw_tracks


WINDOW_NAME = "Body Crop YOLO Pose Preview"


@dataclass
class CameraState:
    camera_index: int
    device_id: str
    device: dai.Device
    pipeline: dai.Pipeline
    queue: dai.MessageQueue
    tracker: object
    last_preview: np.ndarray | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO pose on tracked person crops from one or more OAK cameras."
    )
    parser.add_argument("--device-id", nargs="+", default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--model", type=Path, default=DEFAULT_PERSON_DETECTOR_MODEL)
    parser.add_argument("--input-width", type=int, default=DEFAULT_DETECTION_INPUT_WIDTH)
    parser.add_argument("--input-height", type=int, default=DEFAULT_DETECTION_INPUT_HEIGHT)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD)
    parser.add_argument("--nms-threshold", type=float, default=DEFAULT_DETECTION_NMS_THRESHOLD)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--pose-score-threshold", type=float, default=DEFAULT_POSE_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--pose-keypoint-threshold",
        type=float,
        default=DEFAULT_POSE_KEYPOINT_THRESHOLD,
    )
    parser.add_argument("--pose-nms-threshold", type=float, default=0.45)
    parser.add_argument("--pose-crop-margin", type=float, default=0.05)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--display-columns", type=int, default=2)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Frame dimensions and FPS must be positive.")
    if args.pose_crop_margin < 0.0:
        raise ValueError("--pose-crop-margin must not be negative.")
    for name in (
        "score_threshold",
        "nms_threshold",
        "pose_score_threshold",
        "pose_keypoint_threshold",
        "pose_nms_threshold",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between zero and one.")


def _resolve_device_ids(requested: Sequence[str] | None) -> list[str]:
    available_ids = [device_identifier(info) for info in list_available_devices()]
    if requested is None:
        if not available_ids:
            raise RuntimeError("No OAK devices found.")
        return available_ids
    missing = [device_id for device_id in requested if device_id not in available_ids]
    if missing:
        raise RuntimeError(
            f"Requested device ids not found: {', '.join(missing)}. "
            f"Available device ids: {', '.join(available_ids) or 'none'}"
        )
    return list(requested)


def _open_camera(
    stack: ExitStack,
    *,
    camera_index: int,
    device_id: str,
    args: argparse.Namespace,
) -> CameraState:
    device = dai.Device(device_id)
    configure_live_device(device)
    print_connected_device(device)
    pipeline = stack.enter_context(dai.Pipeline(device))
    camera = pipeline.create(dai.node.Camera).build()
    output = camera.requestOutput(
        size=(args.width, args.height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=args.fps,
    )
    queue = output.createOutputQueue(maxSize=2, blocking=False)
    pipeline.start()
    tracker_args = argparse.Namespace(
        tracker_backend=DEFAULT_PERSON_TRACKER_BACKEND,
        iou_threshold=DEFAULT_TRACKING_IOU_THRESHOLD,
        max_missed=DEFAULT_TRACKING_MAX_MISSED,
    )
    print(f"Started pose preview camera {camera_index + 1}: {device_id}")
    return CameraState(
        camera_index=camera_index,
        device_id=device_id,
        device=device,
        pipeline=pipeline,
        queue=queue,
        tracker=build_person_tracker(tracker_args),
    )


def _drain_latest(queue: dai.MessageQueue) -> dai.ImgFrame | None:
    latest = queue.tryGet()
    newer = queue.tryGet()
    while newer is not None:
        latest = newer
        newer = queue.tryGet()
    return latest


def _mosaic(states: Sequence[CameraState], args: argparse.Namespace) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for state in states:
        frame = state.last_preview
        if frame is None:
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Camera {state.camera_index + 1}: waiting",
                (20, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
        tiles.append(
            cv2.resize(
                frame,
                (args.tile_width, args.tile_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    columns = min(args.display_columns, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    blank = np.zeros_like(tiles[0])
    tiles.extend(blank.copy() for _ in range(rows * columns - len(tiles)))
    return np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )


def main() -> None:
    args = build_argparser().parse_args()
    _validate_args(args)
    if args.list_devices:
        print_available_devices()
        return

    device_ids = _resolve_device_ids(args.device_id)
    person_detector = YoloOnnxPersonDetector(
        model_path=args.model,
        input_size=(args.input_width, args.input_height),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    pose_estimator = YoloOnnxPoseEstimator(
        args.pose_model,
        score_threshold=args.pose_score_threshold,
        keypoint_threshold=args.pose_keypoint_threshold,
        nms_threshold=args.pose_nms_threshold,
    )

    with ExitStack() as stack:
        states = [
            _open_camera(
                stack, camera_index=index, device_id=device_id, args=args
            )
            for index, device_id in enumerate(device_ids)
        ]
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("Body-crop pose preview started. Press q to quit.")
        try:
            while True:
                updated = False
                for state in states:
                    if state.device.isClosed() or not state.pipeline.isRunning():
                        raise RuntimeError(f"Camera stopped: {state.device_id}")
                    message = _drain_latest(state.queue)
                    if message is None:
                        continue
                    updated = True
                    frame = message.getCvFrame()
                    tracks = state.tracker.update(person_detector.detect(frame))
                    preview = frame.copy()
                    draw_tracks(preview, tracks)
                    for track in tracks:
                        if track.status not in {"NEW", "TRACKED"}:
                            continue
                        try:
                            crop, crop_box = crop_person_for_pose(
                                frame,
                                (track.x1, track.y1, track.x2, track.y2),
                                margin_fraction=args.pose_crop_margin,
                            )
                        except ValueError:
                            continue
                        started = time.monotonic()
                        detection = pose_estimator.estimate(crop)
                        inference_ms = int(round((time.monotonic() - started) * 1000))
                        if detection is None:
                            continue
                        mapped = map_pose_detection_to_source(
                            detection,
                            crop_box=crop_box,
                            source_width=frame.shape[1],
                            source_height=frame.shape[0],
                        )
                        observation = PoseObservation(
                            camera_index=state.camera_index,
                            device_id=state.device_id,
                            track_id=track.track_id,
                            rgb_sequence_number=int(message.getSequenceNum()),
                            host_synced_seconds=float(
                                message.getTimestamp().total_seconds()
                            ),
                            observed_at_unix_milliseconds=time.time_ns() // 1_000_000,
                            inference_milliseconds=inference_ms,
                            person_box=(track.x1, track.y1, track.x2, track.y2),
                            pose_box=mapped.bounding_box,
                            landmarks=mapped.landmarks,
                        )
                        draw_pose_observation(
                            preview,
                            observation,
                            keypoint_threshold=args.pose_keypoint_threshold,
                        )
                        cv2.putText(
                            preview,
                            f"T{track.track_id} pose {inference_ms}ms",
                            (track.x1, max(24, track.y1 - 28)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 220, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    state.last_preview = preview

                cv2.imshow(WINDOW_NAME, _mosaic(states, args))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if not updated:
                    time.sleep(0.005)
        except KeyboardInterrupt:
            print("Body-crop pose preview interrupted.")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
