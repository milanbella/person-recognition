from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from pipeline.camera import add_device_args
from pipeline.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_PERSON_DETECTOR_BACKEND,
    DEFAULT_PERSON_DETECTOR_MODEL,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)
from pipeline.onnx_runtime import prepare_onnx_runtime


DETECTOR_BACKEND_CHOICES = ["yolo"]


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str = "person"


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...


def build_detection_argparser(
    description: str = "Host-side YOLO person detection on OAK USB frames.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_detection_args(parser)
    return parser


def add_detection_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_device_args(parser)
    parser.add_argument(
        "--detector-backend",
        choices=DETECTOR_BACKEND_CHOICES,
        default=DEFAULT_PERSON_DETECTOR_BACKEND,
        help="Person detector backend. YOLO is the only supported backend.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_PERSON_DETECTOR_MODEL,
        help="Path to the host-side person detector ONNX model.",
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
        "--yolo-person-class-id",
        type=int,
        default=0,
        help="YOLO class id representing a person. COCO models use class 0.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera output FPS.",
    )
    return parser


def build_person_detector(args: argparse.Namespace) -> PersonDetector:
    backend = getattr(args, "detector_backend", DEFAULT_PERSON_DETECTOR_BACKEND)
    common_args = {
        "model_path": args.model,
        "input_size": (args.input_width, args.input_height),
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
    }
    if backend == "yolo":
        return YoloOnnxPersonDetector(
            **common_args,
            person_class_id=getattr(args, "yolo_person_class_id", 0),
        )
    raise ValueError(f"Unsupported detector backend: {backend}")


def letterbox_yolo_frame(
    frame: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[float, float]]:
    input_width, input_height = input_size
    frame_height, frame_width = frame.shape[:2]
    scale = min(input_width / frame_width, input_height / frame_height)
    resized_width = max(1, int(round(frame_width * scale)))
    resized_height = max(1, int(round(frame_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_width - resized_width) / 2.0
    pad_y = (input_height - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, (float(left), float(top))


def decode_yolo_person_output(
    output: np.ndarray,
    *,
    person_class_id: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(output)
    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise ValueError(f"YOLO batch output must contain exactly one frame: {rows.shape}")
        rows = rows[0]
    if rows.ndim != 2:
        raise ValueError(f"Unsupported YOLO output shape: {np.asarray(output).shape}")
    known_feature_counts = {6, 84, 85}
    if (
        rows.shape[0] in known_feature_counts
        and rows.shape[1] not in known_feature_counts
    ) or (
        rows.shape[0] <= 512
        and rows.shape[1] > rows.shape[0]
        and rows.shape[1] not in known_feature_counts
    ):
        rows = rows.T
    if rows.shape[1] < 5:
        raise ValueError(f"YOLO output must have at least 5 values per candidate: {rows.shape}")

    boxes: list[list[float]] = []
    scores: list[float] = []
    feature_count = rows.shape[1]
    end_to_end = feature_count == 6 and bool(
        np.all(np.abs(rows[:, 5] - np.round(rows[:, 5])) < 1e-3)
    )

    for row in rows:
        if end_to_end:
            if int(round(float(row[5]))) != person_class_id:
                continue
            score = float(row[4])
            x1, y1, x2, y2 = (float(value) for value in row[:4])
        else:
            if feature_count == 85:
                class_index = 5 + person_class_id
                if class_index >= feature_count:
                    continue
                score = float(row[4]) * float(row[class_index])
            else:
                class_index = 4 + person_class_id
                if class_index >= feature_count:
                    continue
                score = float(row[class_index])
            center_x, center_y, width, height = (float(value) for value in row[:4])
            x1 = center_x - width / 2.0
            y1 = center_y - height / 2.0
            x2 = center_x + width / 2.0
            y2 = center_y + height / 2.0

        if score < score_threshold or x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(score)

    return (
        np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        np.asarray(scores, dtype=np.float32),
    )


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_width = np.maximum(0.0, np.minimum(x2[current], x2[remaining]) - np.maximum(x1[current], x1[remaining]))
        intersection_height = np.maximum(0.0, np.minimum(y2[current], y2[remaining]) - np.maximum(y1[current], y1[remaining]))
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return keep


class YoloOnnxPersonDetector:
    def __init__(
        self,
        model_path: Path,
        input_size: tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
        person_class_id: int = 0,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found: {model_path}")
        if model_path.suffix.lower() != ".onnx":
            raise ValueError("YOLO backend requires an ONNX model. Export the model before use.")
        if person_class_id < 0:
            raise ValueError("YOLO person class id cannot be negative.")

        self.model_path = model_path
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.person_class_id = person_class_id
        available = prepare_onnx_runtime()
        requested_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        try:
            self.session = ort.InferenceSession(str(model_path), providers=requested_providers)
        except Exception as exc:
            if requested_providers == ["CPUExecutionProvider"]:
                raise
            print(f"CUDAExecutionProvider unavailable for YOLO, falling back to CPU: {exc}")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_inputs = self.session.get_inputs()
        if len(model_inputs) != 1:
            raise ValueError(f"YOLO ONNX model must have exactly one image input, found {len(model_inputs)}.")
        model_input = model_inputs[0]
        self.input_name = model_input.name
        model_shape = model_input.shape
        if len(model_shape) != 4:
            raise ValueError(f"YOLO ONNX input must be NCHW, found shape {model_shape}.")
        model_height, model_width = model_shape[-2], model_shape[-1]
        requested_width, requested_height = input_size
        if isinstance(model_width, int) and model_width != requested_width:
            raise ValueError(
                f"YOLO model input width is {model_width}, but --input-width is {requested_width}."
            )
        if isinstance(model_height, int) and model_height != requested_height:
            raise ValueError(
                f"YOLO model input height is {model_height}, but --input-height is {requested_height}."
            )
        applied = self.session.get_providers()
        provider_name = "CUDAExecutionProvider" if "CUDAExecutionProvider" in applied else "CPUExecutionProvider"
        print(f"Using ONNX Runtime {provider_name} for YOLO.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        tensor, scale, padding = letterbox_yolo_frame(frame, self.input_size)
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            return []
        boxes, scores = decode_yolo_person_output(
            outputs[0],
            person_class_id=self.person_class_id,
            score_threshold=self.score_threshold,
        )
        keep = nms_xyxy(boxes, scores, self.nms_threshold)
        frame_height, frame_width = frame.shape[:2]
        pad_x, pad_y = padding
        detections: list[Detection] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            x1 = float(np.clip((x1 - pad_x) / scale, 0, frame_width - 1))
            y1 = float(np.clip((y1 - pad_y) / scale, 0, frame_height - 1))
            x2 = float(np.clip((x2 - pad_x) / scale, 0, frame_width))
            y2 = float(np.clip((y2 - pad_y) / scale, 0, frame_height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    score=float(scores[index]),
                )
            )
        return detections


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{detection.label} {detection.score:.2f}",
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
