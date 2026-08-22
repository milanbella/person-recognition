from __future__ import annotations

import argparse
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
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
from pipeline.detection import build_person_detector
from pipeline.product_detection import (
    DEFAULT_PRODUCT_MODEL,
    DEFAULT_PRODUCT_SCORE_THRESHOLD,
    ProductDetection,
    YoloOnnxProductDetector,
    expanded_person_crop,
    product_center_matches_person,
    translate_product_detection,
)
from pipeline.tracking import Track, build_person_tracker, draw_tracks


WINDOW_NAME = "Body Crop Product Detection"


@dataclass
class CameraState:
    camera_index: int
    device_id: str
    device: dai.Device
    pipeline: dai.Pipeline
    queue: dai.MessageQueue
    tracker: object
    last_preview: np.ndarray | None = None
    last_summary: tuple[tuple[str, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CropJob:
    camera: CameraState
    track: Track
    crop: np.ndarray
    crop_box: tuple[int, int, int, int]
    person_box_in_crop: tuple[int, int, int, int]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview multiple OAK cameras, track people, and run product YOLO "
            "on expanded body crops."
        )
    )
    parser.add_argument(
        "--device-id",
        nargs="+",
        help="OAK device IDs to preview. Defaults to every available device.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=min(DEFAULT_CAMERA_FPS, 15))

    parser.add_argument(
        "--detector-backend",
        choices=["yolo"],
        default=DEFAULT_PERSON_DETECTOR_BACKEND,
    )
    parser.add_argument(
        "--person-model", dest="model", type=Path, default=DEFAULT_PERSON_DETECTOR_MODEL
    )
    parser.add_argument(
        "--person-input-width",
        dest="input_width",
        type=int,
        default=DEFAULT_DETECTION_INPUT_WIDTH,
    )
    parser.add_argument(
        "--person-input-height",
        dest="input_height",
        type=int,
        default=DEFAULT_DETECTION_INPUT_HEIGHT,
    )
    parser.add_argument(
        "--person-score-threshold",
        dest="score_threshold",
        type=float,
        default=DEFAULT_DETECTION_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--person-nms-threshold",
        dest="nms_threshold",
        type=float,
        default=DEFAULT_DETECTION_NMS_THRESHOLD,
    )
    parser.add_argument("--yolo-person-class-id", type=int, default=0)

    parser.add_argument(
        "--tracker-backend",
        choices=["iou"],
        default=DEFAULT_PERSON_TRACKER_BACKEND,
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=DEFAULT_TRACKING_IOU_THRESHOLD
    )
    parser.add_argument("--max-missed", type=int, default=DEFAULT_TRACKING_MAX_MISSED)

    parser.add_argument("--product-model", type=Path, default=DEFAULT_PRODUCT_MODEL)
    parser.add_argument(
        "--product-score-threshold",
        type=float,
        default=DEFAULT_PRODUCT_SCORE_THRESHOLD,
    )
    parser.add_argument("--product-nms-threshold", type=float, default=0.45)
    parser.add_argument("--crop-margin", type=float, default=0.35)
    parser.add_argument(
        "--association-margin",
        type=float,
        default=0.10,
        help="Extra fraction around the person box accepted for product centers.",
    )

    parser.add_argument("--display-columns", type=int, default=2)
    parser.add_argument("--tile-width", type=int, default=720)
    parser.add_argument("--tile-height", type=int, default=405)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "--width": args.width,
        "--height": args.height,
        "--fps": args.fps,
        "--person-input-width": args.input_width,
        "--person-input-height": args.input_height,
        "--display-columns": args.display_columns,
        "--tile-width": args.tile_width,
        "--tile-height": args.tile_height,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Values must be positive: {', '.join(invalid)}")
    for name, value in (
        ("--person-score-threshold", args.score_threshold),
        ("--product-score-threshold", args.product_score_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one.")
    if args.crop_margin < 0.0 or args.association_margin < 0.0:
        raise ValueError("Crop and association margins must not be negative.")


def _resolve_device_ids(requested: Sequence[str] | None) -> list[str]:
    available = list_available_devices()
    available_ids = [device_identifier(info) for info in available]
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
    print(f"Started camera {camera_index + 1}: {device_id}")
    return CameraState(
        camera_index=camera_index,
        device_id=device_id,
        device=device,
        pipeline=pipeline,
        queue=queue,
        tracker=build_person_tracker(args),
    )


def _drain_latest(queue: dai.MessageQueue) -> dai.ImgFrame | None:
    latest = queue.tryGet()
    newer = queue.tryGet()
    while newer is not None:
        latest = newer
        newer = queue.tryGet()
    return latest


def _draw_product(
    frame: np.ndarray,
    detection: ProductDetection,
    *,
    track_id: int,
) -> None:
    color = (
        64 + (detection.class_id * 47) % 192,
        64 + (detection.class_id * 89) % 192,
        64 + (detection.class_id * 131) % 192,
    )
    cv2.rectangle(
        frame,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        color,
        3,
    )
    cv2.putText(
        frame,
        f"T{track_id} {detection.label} {detection.score:.2f}",
        (detection.x1, max(24, detection.y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_camera_header(frame: np.ndarray, state: CameraState) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Camera {state.camera_index + 1}  {state.device_id}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _blank_tile(state: CameraState, args: argparse.Namespace) -> np.ndarray:
    tile = np.zeros((args.tile_height, args.tile_width, 3), dtype=np.uint8)
    cv2.putText(
        tile,
        f"Camera {state.camera_index + 1}: waiting for frame",
        (20, args.tile_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return tile


def _build_mosaic(states: Sequence[CameraState], args: argparse.Namespace) -> np.ndarray:
    tiles = [
        cv2.resize(
            state.last_preview
            if state.last_preview is not None
            else _blank_tile(state, args),
            (args.tile_width, args.tile_height),
            interpolation=cv2.INTER_AREA,
        )
        for state in states
    ]
    columns = min(args.display_columns, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    blank = np.zeros_like(tiles[0])
    tiles.extend(blank.copy() for _ in range(rows * columns - len(tiles)))
    return np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )


def _run_product_jobs(
    jobs: Sequence[CropJob],
    detector: YoloOnnxProductDetector,
    args: argparse.Namespace,
) -> dict[int, list[tuple[CropJob, ProductDetection]]]:
    by_camera: dict[int, list[tuple[CropJob, ProductDetection]]] = {}
    for start in range(0, len(jobs), detector.batch_size):
        batch = jobs[start : start + detector.batch_size]
        detections_by_crop = detector.detect_batch([job.crop for job in batch])
        for job, detections in zip(batch, detections_by_crop):
            for detection in detections:
                if not product_center_matches_person(
                    detection,
                    job.person_box_in_crop,
                    margin_fraction=args.association_margin,
                ):
                    continue
                translated = translate_product_detection(detection, job.crop_box)
                by_camera.setdefault(job.camera.camera_index, []).append(
                    (job, translated)
                )
    return by_camera


def main() -> None:
    args = build_argparser().parse_args()
    _validate_args(args)
    if args.list_devices:
        print_available_devices()
        return

    device_ids = _resolve_device_ids(args.device_id)
    person_detector = build_person_detector(args)
    product_detector = YoloOnnxProductDetector(
        args.product_model,
        score_threshold=args.product_score_threshold,
        nms_threshold=args.product_nms_threshold,
    )

    with ExitStack() as stack:
        states: list[CameraState] = []
        for camera_index, device_id in enumerate(device_ids):
            states.append(
                _open_camera(
                    stack,
                    camera_index=camera_index,
                    device_id=device_id,
                    args=args,
                )
            )
            time.sleep(0.5)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("Body-crop product preview started. Press q to quit.")
        try:
            while states:
                jobs: list[CropJob] = []
                updated: list[CameraState] = []
                for state in states:
                    if state.device.isClosed() or not state.pipeline.isRunning():
                        raise RuntimeError(f"Camera stopped: {state.device_id}")
                    message = _drain_latest(state.queue)
                    if message is None:
                        continue
                    frame = message.getCvFrame()
                    detections = person_detector.detect(frame)
                    tracks = state.tracker.update(detections)
                    preview = frame.copy()
                    draw_tracks(preview, tracks)
                    for track in tracks:
                        if track.status not in {"NEW", "TRACKED"}:
                            continue
                        crop, crop_box, person_box = expanded_person_crop(
                            frame,
                            (track.x1, track.y1, track.x2, track.y2),
                            margin_fraction=args.crop_margin,
                        )
                        jobs.append(
                            CropJob(
                                camera=state,
                                track=track,
                                crop=crop,
                                crop_box=crop_box,
                                person_box_in_crop=person_box,
                            )
                        )
                        cv2.rectangle(
                            preview,
                            crop_box[:2],
                            crop_box[2:],
                            (255, 0, 255),
                            2,
                        )
                    state.last_preview = preview
                    updated.append(state)

                products_by_camera = _run_product_jobs(jobs, product_detector, args)
                for state in updated:
                    assert state.last_preview is not None
                    products = products_by_camera.get(state.camera_index, [])
                    counts: dict[str, int] = {}
                    for job, detection in products:
                        counts[detection.label] = counts.get(detection.label, 0) + 1
                        _draw_product(
                            state.last_preview,
                            detection,
                            track_id=job.track.track_id,
                        )
                    summary = tuple(sorted(counts.items()))
                    if summary != state.last_summary:
                        print(
                            f"Camera {state.camera_index + 1} "
                            f"device_id={state.device_id} products={dict(summary)}"
                        )
                        state.last_summary = summary
                    _draw_camera_header(state.last_preview, state)

                cv2.imshow(WINDOW_NAME, _build_mosaic(states, args))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if not updated:
                    time.sleep(0.005)
        except KeyboardInterrupt:
            print("Body-crop product preview interrupted.")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
