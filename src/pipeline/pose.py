from __future__ import annotations
from pipeline.camera_logging import camera_log_fields

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from pipeline.detection import letterbox_yolo_frame, nms_xyxy
from pipeline.onnx_runtime import prepare_onnx_runtime


DEFAULT_POSE_MODEL = (
    Path(__file__).resolve().parent.parent.parent / "models" / "yolo26n-pose.onnx"
)
DEFAULT_POSE_SCORE_THRESHOLD = 0.50
DEFAULT_POSE_KEYPOINT_THRESHOLD = 0.35
COCO_POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_POSE_CONNECTIONS = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class PoseLandmark:
    name: str
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class PoseDetection:
    bounding_box: tuple[int, int, int, int]
    score: float
    landmarks: tuple[PoseLandmark, ...]


@dataclass(frozen=True)
class PoseRequest:
    camera_index: int
    device_id: str
    track_id: int
    rgb_sequence_number: int
    host_synced_seconds: float
    submitted_at_unix_milliseconds: int
    source_frame_width: int
    source_frame_height: int
    crop_box: tuple[int, int, int, int]
    person_box: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class PoseObservation:
    camera_index: int
    device_id: str
    track_id: int
    rgb_sequence_number: int
    host_synced_seconds: float
    observed_at_unix_milliseconds: int
    inference_milliseconds: int
    person_box: tuple[int, int, int, int]
    pose_box: tuple[int, int, int, int]
    landmarks: tuple[PoseLandmark, ...]
    queue_age_milliseconds: int = 0
    source_frame_width: int = 0
    source_frame_height: int = 0


def decode_yolo_pose_output(
    output: np.ndarray,
    *,
    score_threshold: float,
    keypoint_count: int = len(COCO_POSE_LANDMARK_NAMES),
    batch_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw or end-to-end Ultralytics pose rows in model coordinates."""
    rows = np.asarray(output)
    if rows.ndim == 3:
        if batch_index < 0 or batch_index >= rows.shape[0]:
            raise ValueError(
                f"Pose batch index {batch_index} is outside output shape {rows.shape}."
            )
        rows = rows[batch_index]
    if rows.ndim != 2:
        raise ValueError(f"Unsupported pose output shape: {np.asarray(output).shape}")

    raw_features = 5 + keypoint_count * 3
    end_to_end_features = 6 + keypoint_count * 3
    if rows.shape[0] in {raw_features, end_to_end_features} and rows.shape[1] not in {
        raw_features,
        end_to_end_features,
    }:
        rows = rows.T
    if rows.shape[1] not in {raw_features, end_to_end_features}:
        raise ValueError(
            "Pose output must contain box, confidence, and "
            f"{keypoint_count} x/y/score keypoints; found {rows.shape}."
        )

    end_to_end = rows.shape[1] == end_to_end_features
    boxes: list[list[float]] = []
    scores: list[float] = []
    keypoints: list[np.ndarray] = []
    for row in rows:
        score = float(row[4])
        if score < score_threshold:
            continue
        if end_to_end:
            class_id = int(round(float(row[5])))
            if class_id != 0:
                continue
            x1, y1, x2, y2 = (float(value) for value in row[:4])
            offset = 6
        else:
            center_x, center_y, width, height = (float(value) for value in row[:4])
            x1 = center_x - width / 2.0
            y1 = center_y - height / 2.0
            x2 = center_x + width / 2.0
            y2 = center_y + height / 2.0
            offset = 5
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        keypoints.append(
            np.asarray(row[offset:], dtype=np.float32).reshape((keypoint_count, 3))
        )

    return (
        np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        np.asarray(scores, dtype=np.float32),
        np.asarray(keypoints, dtype=np.float32).reshape((-1, keypoint_count, 3)),
    )


def map_pose_detection_to_source(
    detection: PoseDetection,
    *,
    crop_box: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
) -> PoseDetection:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Pose source dimensions must be positive.")
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_box
    pose_x1, pose_y1, pose_x2, pose_y2 = detection.bounding_box
    pose_box = (
        crop_x1 + pose_x1,
        crop_y1 + pose_y1,
        crop_x1 + pose_x2,
        crop_y1 + pose_y2,
    )
    landmarks = tuple(
        PoseLandmark(
            name=landmark.name,
            x=float(np.clip((crop_x1 + landmark.x) / source_width, 0.0, 1.0)),
            y=float(np.clip((crop_y1 + landmark.y) / source_height, 0.0, 1.0)),
            score=landmark.score,
        )
        for landmark in detection.landmarks
    )
    return PoseDetection(
        bounding_box=pose_box,
        score=detection.score,
        landmarks=landmarks,
    )


class YoloOnnxPoseEstimator:
    def __init__(
        self,
        model_path: Path,
        *,
        score_threshold: float = DEFAULT_POSE_SCORE_THRESHOLD,
        keypoint_threshold: float = DEFAULT_POSE_KEYPOINT_THRESHOLD,
        nms_threshold: float = 0.45,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")
        if model_path.suffix.lower() != ".onnx":
            raise ValueError("YOLO pose backend requires an ONNX model.")
        available = prepare_onnx_runtime()
        requested = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        try:
            self.session = ort.InferenceSession(str(model_path), providers=requested)
        except Exception as exc:
            if requested == ["CPUExecutionProvider"]:
                raise
            print(
                "CUDAExecutionProvider unavailable for pose YOLO, "
                f"falling back to CPU: {exc}"
            )
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError("Pose model must have one NCHW image input.")
        batch_size, _channels, input_height, input_width = inputs[0].shape
        if not all(
            isinstance(value, int)
            for value in (batch_size, input_height, input_width)
        ):
            raise ValueError("Pose runtime currently requires a fixed-shape ONNX model.")
        if int(batch_size) <= 0:
            raise ValueError("Pose ONNX batch size must be positive.")
        self.input_name = inputs[0].name
        self.batch_size = int(batch_size)
        self.input_size = (int(input_width), int(input_height))
        self.score_threshold = score_threshold
        self.keypoint_threshold = keypoint_threshold
        self.nms_threshold = nms_threshold
        provider = self.session.get_providers()[0]
        print(
            f"Loaded pose YOLO model={model_path} provider={provider} "
            f"input={self.input_size[0]}x{self.input_size[1]}"
        )

    def estimate(self, frame: np.ndarray) -> PoseDetection | None:
        tensor, scale, (pad_x, pad_y) = letterbox_yolo_frame(frame, self.input_size)
        if self.batch_size > 1:
            tensor = np.repeat(tensor, self.batch_size, axis=0)
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            return None
        boxes, scores, keypoints = decode_yolo_pose_output(
            outputs[0], score_threshold=self.score_threshold
        )
        keep = nms_xyxy(boxes, scores, self.nms_threshold)
        if not keep:
            return None
        frame_height, frame_width = frame.shape[:2]
        frame_center = np.asarray([frame_width / 2.0, frame_height / 2.0])

        candidates: list[tuple[float, PoseDetection]] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            mapped_box = (
                int(round(np.clip((x1 - pad_x) / scale, 0, frame_width - 1))),
                int(round(np.clip((y1 - pad_y) / scale, 0, frame_height - 1))),
                int(round(np.clip((x2 - pad_x) / scale, 0, frame_width))),
                int(round(np.clip((y2 - pad_y) / scale, 0, frame_height))),
            )
            landmarks = tuple(
                PoseLandmark(
                    name=name,
                    x=float(np.clip((point[0] - pad_x) / scale, 0, frame_width)),
                    y=float(np.clip((point[1] - pad_y) / scale, 0, frame_height)),
                    score=float(point[2]),
                )
                for name, point in zip(COCO_POSE_LANDMARK_NAMES, keypoints[index])
            )
            center = np.asarray(
                [(mapped_box[0] + mapped_box[2]) / 2.0, (mapped_box[1] + mapped_box[3]) / 2.0]
            )
            center_penalty = float(np.linalg.norm(center - frame_center)) / max(
                frame_width, frame_height
            )
            candidates.append(
                (
                    float(scores[index]) - center_penalty,
                    PoseDetection(mapped_box, float(scores[index]), landmarks),
                )
            )
        return max(candidates, key=lambda item: item[0])[1]


class PoseEstimationWorker:
    def __init__(
        self,
        estimator: YoloOnnxPoseEstimator,
        *,
        scan_interval_seconds: float,
        log_results: bool = False,
        inference_lock: threading.Lock | None = None,
    ) -> None:
        self.estimator = estimator
        self.scan_interval_seconds = scan_interval_seconds
        self.log_results = log_results
        self._inference_lock = inference_lock or threading.Lock()
        self._condition = threading.Condition()
        self._pending: dict[tuple[int, int], PoseRequest] = {}
        self._results: deque[PoseObservation] = deque()
        self._last_submitted: dict[tuple[int, int], float] = {}
        self._stopping = False
        self._pending_replacements = 0
        self._thread = threading.Thread(
            target=self._run, name="pose-estimation-worker", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def is_due(self, camera_index: int, track_id: int, host_seconds: float) -> bool:
        with self._condition:
            previous = self._last_submitted.get((camera_index, track_id))
        return previous is None or host_seconds - previous >= self.scan_interval_seconds

    def submit(self, request: PoseRequest) -> bool:
        key = (request.camera_index, request.track_id)
        with self._condition:
            previous = self._last_submitted.get(key)
            if (
                previous is not None
                and request.host_synced_seconds - previous < self.scan_interval_seconds
            ):
                return False
            self._last_submitted[key] = request.host_synced_seconds
            if key in self._pending:
                self._pending_replacements += 1
            self._pending[key] = request
            self._condition.notify()
        return True

    def drain_results(self) -> tuple[PoseObservation, ...]:
        with self._condition:
            results = tuple(self._results)
            self._results.clear()
        return results

    def drain_metrics(self) -> tuple[int, int]:
        with self._condition:
            pending = len(self._pending)
            replacements = self._pending_replacements
            self._pending_replacements = 0
        return pending, replacements

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._stopping)
                if self._stopping and not self._pending:
                    return
                _key, request = self._pending.popitem()
            queue_age_ms = max(
                0,
                time.time_ns() // 1_000_000
                - request.submitted_at_unix_milliseconds,
            )
            started = time.monotonic()
            with self._inference_lock:
                detection = self.estimator.estimate(request.crop)
            inference_ms = int(round((time.monotonic() - started) * 1000.0))
            if detection is None:
                continue
            mapped = map_pose_detection_to_source(
                detection,
                crop_box=request.crop_box,
                source_width=request.source_frame_width,
                source_height=request.source_frame_height,
            )
            observation = PoseObservation(
                camera_index=request.camera_index,
                device_id=request.device_id,
                track_id=request.track_id,
                rgb_sequence_number=request.rgb_sequence_number,
                host_synced_seconds=request.host_synced_seconds,
                observed_at_unix_milliseconds=time.time_ns() // 1_000_000,
                inference_milliseconds=inference_ms,
                person_box=request.person_box,
                pose_box=mapped.bounding_box,
                landmarks=mapped.landmarks,
                queue_age_milliseconds=queue_age_ms,
                source_frame_width=request.source_frame_width,
                source_frame_height=request.source_frame_height,
            )
            if self.log_results:
                visible = sum(
                    landmark.score >= self.estimator.keypoint_threshold
                    for landmark in observation.landmarks
                )
                print(
                    f"POSE_TRACE {camera_log_fields(observation.device_id)} "
                    f"track_id={observation.track_id} "
                    f"sequence={observation.rgb_sequence_number} "
                    f"visible_landmarks={visible} inference_ms={inference_ms} "
                    f"queue_age_ms={queue_age_ms}"
                )
            with self._condition:
                self._results.append(observation)


def crop_person_for_pose(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    *,
    margin_fraction: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if margin_fraction < 0.0:
        raise ValueError("Pose crop margin must not be negative.")
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bounding_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = int(round(width * margin_fraction))
    margin_y = int(round(height * margin_fraction))
    crop_box = (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(frame_width, x2 + margin_x),
        min(frame_height, y2 + margin_y),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"Invalid pose person bounding box: {bounding_box}")
    return frame[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]].copy(), crop_box


def draw_pose_observation(
    frame: np.ndarray,
    observation: PoseObservation,
    *,
    keypoint_threshold: float = DEFAULT_POSE_KEYPOINT_THRESHOLD,
) -> None:
    points: list[tuple[int, int] | None] = []
    for landmark in observation.landmarks:
        if landmark.score < keypoint_threshold:
            points.append(None)
            continue
        point = (
            int(round(landmark.x * frame.shape[1])),
            int(round(landmark.y * frame.shape[0])),
        )
        points.append(point)
        cv2.circle(frame, point, 4, (0, 220, 255), -1, cv2.LINE_AA)
    for start, end in COCO_POSE_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], (0, 180, 255), 2, cv2.LINE_AA)
