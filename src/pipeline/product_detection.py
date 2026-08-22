from __future__ import annotations

import ast
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from pipeline.detection import letterbox_yolo_frame, nms_xyxy
from pipeline.onnx_runtime import prepare_onnx_runtime


DEFAULT_PRODUCT_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "best.onnx"
DEFAULT_PRODUCT_SCORE_THRESHOLD = 0.55


@dataclass(frozen=True)
class ProductDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int
    label: str


@dataclass(frozen=True)
class ProductCropRequest:
    camera_index: int
    device_id: str
    track_id: int | None
    scope: str
    rgb_sequence_number: int
    host_synced_seconds: float
    submitted_at_unix_milliseconds: int
    crop_box: tuple[int, int, int, int]
    person_box_in_crop: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class ProductRecognitionResult:
    camera_index: int
    device_id: str
    track_id: int | None
    scope: str
    rgb_sequence_number: int
    host_synced_seconds: float
    observed_at_unix_milliseconds: int
    inference_milliseconds: int
    crop_box: tuple[int, int, int, int]
    person_box_in_crop: tuple[int, int, int, int]
    detections: tuple[ProductDetection, ...]
    crop_jpeg: bytes
    source_crop_image: bytes | None = None


def encode_lossless_product_crop(crop: np.ndarray) -> bytes:
    encoded, png = cv2.imencode(".png", crop)
    if not encoded:
        raise RuntimeError("Could not encode lossless product source crop.")
    return png.tobytes()


def decode_product_crop(source_crop_image: bytes) -> np.ndarray:
    encoded = np.frombuffer(source_crop_image, dtype=np.uint8)
    crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if crop is None:
        raise ValueError("Frozen product source crop is not a valid image.")
    return crop


def expanded_person_crop(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    *,
    margin_fraction: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if margin_fraction < 0.0:
        raise ValueError("Product crop margin must not be negative.")
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bounding_box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = int(round(box_width * margin_fraction))
    margin_y = int(round(box_height * margin_fraction))
    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(frame_width, x2 + margin_x)
    crop_y2 = min(frame_height, y2 + margin_y)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(f"Invalid person bounding box: {bounding_box}")
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return (
        crop,
        (crop_x1, crop_y1, crop_x2, crop_y2),
        (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1),
    )


def translate_product_detection(
    detection: ProductDetection,
    crop_box: tuple[int, int, int, int],
) -> ProductDetection:
    """Map a crop-relative product detection back to source-frame coordinates."""
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_box
    return ProductDetection(
        x1=detection.x1 + crop_x1,
        y1=detection.y1 + crop_y1,
        x2=detection.x2 + crop_x1,
        y2=detection.y2 + crop_y1,
        score=detection.score,
        class_id=detection.class_id,
        label=detection.label,
    )


def parse_yolo_class_names(metadata: Mapping[str, str]) -> dict[int, str]:
    raw_names = metadata.get("names")
    if not raw_names:
        raise ValueError("YOLO model metadata does not contain class names.")
    parsed = ast.literal_eval(raw_names)
    if not isinstance(parsed, dict):
        raise ValueError("YOLO model class names metadata must be a dictionary.")
    return {int(class_id): str(label) for class_id, label in parsed.items()}


def decode_end_to_end_product_output(
    output: np.ndarray,
    *,
    class_names: Mapping[int, str],
    score_threshold: float,
    batch_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(output)
    if rows.ndim == 3:
        if batch_index < 0 or batch_index >= rows.shape[0]:
            raise ValueError(
                f"YOLO batch index {batch_index} is outside output shape {rows.shape}."
            )
        rows = rows[batch_index]
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(
            "Product model must return end-to-end rows "
            f"[x1, y1, x2, y2, score, class_id], found {rows.shape}."
        )

    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for row in rows:
        score = float(row[4])
        class_id = int(round(float(row[5])))
        x1, y1, x2, y2 = (float(value) for value in row[:4])
        if (
            score < score_threshold
            or class_id not in class_names
            or x2 <= x1
            or y2 <= y1
        ):
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)
    return (
        np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        np.asarray(scores, dtype=np.float32),
        np.asarray(class_ids, dtype=np.int64),
    )


class YoloOnnxProductDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        score_threshold: float = DEFAULT_PRODUCT_SCORE_THRESHOLD,
        nms_threshold: float = 0.45,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Product model not found: {model_path}")
        available = prepare_onnx_runtime()
        requested_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        try:
            self.session = ort.InferenceSession(
                str(model_path), providers=requested_providers
            )
        except Exception as exc:
            if requested_providers == ["CPUExecutionProvider"]:
                raise
            print(
                "CUDAExecutionProvider unavailable for product YOLO, "
                f"falling back to CPU: {exc}"
            )
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError("Product YOLO model must have one NCHW image input.")
        batch_size, _, input_height, input_width = inputs[0].shape
        if not all(isinstance(value, int) for value in (batch_size, input_height, input_width)):
            raise ValueError("Product preview currently requires a fixed-shape ONNX model.")
        self.input_name = inputs[0].name
        self.batch_size = int(batch_size)
        self.input_size = (int(input_width), int(input_height))
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.class_names = parse_yolo_class_names(
            self.session.get_modelmeta().custom_metadata_map
        )
        provider = self.session.get_providers()[0]
        print(
            f"Loaded product YOLO model={model_path} provider={provider} "
            f"input={self.input_size[0]}x{self.input_size[1]} "
            f"batch={self.batch_size} classes={len(self.class_names)}"
        )

    def detect(self, frame: np.ndarray) -> tuple[ProductDetection, ...]:
        return self.detect_batch((frame,))[0]

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
    ) -> tuple[tuple[ProductDetection, ...], ...]:
        if not frames:
            return ()
        if len(frames) > self.batch_size:
            raise ValueError(
                f"Product model batch accepts at most {self.batch_size} frames."
            )
        prepared = [letterbox_yolo_frame(frame, self.input_size) for frame in frames]
        tensors = [item[0] for item in prepared]
        while len(tensors) < self.batch_size:
            tensors.append(np.zeros_like(tensors[0]))
        tensor = np.concatenate(tensors, axis=0)
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            return tuple(() for _frame in frames)

        results: list[tuple[ProductDetection, ...]] = []
        for batch_index, (frame, (_tensor, scale, (pad_x, pad_y))) in enumerate(
            zip(frames, prepared)
        ):
            boxes, scores, class_ids = decode_end_to_end_product_output(
                outputs[0],
                class_names=self.class_names,
                score_threshold=self.score_threshold,
                batch_index=batch_index,
            )
            keep: list[int] = []
            for class_id in np.unique(class_ids):
                indices = np.flatnonzero(class_ids == class_id)
                selected = nms_xyxy(
                    boxes[indices], scores[indices], self.nms_threshold
                )
                keep.extend(int(indices[index]) for index in selected)

            frame_height, frame_width = frame.shape[:2]
            detections: list[ProductDetection] = []
            for index in sorted(
                keep, key=lambda item: float(scores[item]), reverse=True
            ):
                x1, y1, x2, y2 = boxes[index]
                x1 = float(np.clip((x1 - pad_x) / scale, 0, frame_width - 1))
                y1 = float(np.clip((y1 - pad_y) / scale, 0, frame_height - 1))
                x2 = float(np.clip((x2 - pad_x) / scale, 0, frame_width))
                y2 = float(np.clip((y2 - pad_y) / scale, 0, frame_height))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(class_ids[index])
                detections.append(
                    ProductDetection(
                        x1=int(round(x1)),
                        y1=int(round(y1)),
                        x2=int(round(x2)),
                        y2=int(round(y2)),
                        score=float(scores[index]),
                        class_id=class_id,
                        label=self.class_names[class_id],
                    )
                )
            results.append(tuple(detections))
        return tuple(results)


def draw_product_detections(
    frame: np.ndarray,
    detections: Sequence[ProductDetection],
) -> None:
    for detection in detections:
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
            2,
        )
        cv2.putText(
            frame,
            f"{detection.label} {detection.score:.2f}",
            (detection.x1, max(22, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def product_center_matches_person(
    detection: ProductDetection,
    person_box: tuple[int, int, int, int],
    *,
    margin_fraction: float,
) -> bool:
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = width * margin_fraction
    margin_y = height * margin_fraction
    center_x = (detection.x1 + detection.x2) / 2.0
    center_y = (detection.y1 + detection.y2) / 2.0
    return (
        x1 - margin_x <= center_x <= x2 + margin_x
        and y1 - margin_y <= center_y <= y2 + margin_y
    )


class ProductRecognitionWorker:
    def __init__(
        self,
        detector: YoloOnnxProductDetector,
        *,
        scan_interval_seconds: float,
        association_margin_fraction: float,
        jpeg_quality: int = 80,
        log_results: bool = False,
    ) -> None:
        self.detector = detector
        self.scan_interval_seconds = scan_interval_seconds
        self.association_margin_fraction = association_margin_fraction
        self.jpeg_quality = jpeg_quality
        self.log_results = log_results
        self._condition = threading.Condition()
        self._inference_lock = threading.Lock()
        self._pending: dict[tuple[int, int | None], ProductCropRequest] = {}
        self._results: deque[ProductRecognitionResult] = deque()
        self._last_submitted: dict[tuple[int, int | None], float] = {}
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="product-recognition-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def is_due(
        self,
        camera_index: int,
        track_id: int | None,
        host_synced_seconds: float,
    ) -> bool:
        with self._condition:
            previous = self._last_submitted.get((camera_index, track_id))
        return (
            previous is None
            or host_synced_seconds - previous >= self.scan_interval_seconds
        )

    def submit(self, request: ProductCropRequest) -> bool:
        key = (request.camera_index, request.track_id)
        with self._condition:
            previous = self._last_submitted.get(key)
            if (
                previous is not None
                and request.host_synced_seconds - previous
                < self.scan_interval_seconds
            ):
                return False
            self._last_submitted[key] = request.host_synced_seconds
            self._pending[key] = request
            self._condition.notify()
        return True

    def drain_results(self) -> tuple[ProductRecognitionResult, ...]:
        with self._condition:
            results = tuple(self._results)
            self._results.clear()
        return results

    def redetect_crop_image(
        self,
        source_crop_image: bytes,
        scope: str,
        person_box_in_crop: tuple[int, int, int, int],
    ) -> tuple[tuple[ProductDetection, ...], bytes, int]:
        crop = decode_product_crop(source_crop_image)
        started = time.monotonic()
        with self._inference_lock:
            candidates = self.detector.detect(crop)
        inference_ms = int(round((time.monotonic() - started) * 1000.0))
        associated = (
            tuple(candidates)
            if scope == "full_frame"
            else tuple(
                candidate
                for candidate in candidates
                if product_center_matches_person(
                    candidate,
                    person_box_in_crop,
                    margin_fraction=self.association_margin_fraction,
                )
            )
        )
        annotated = crop.copy()
        draw_product_detections(annotated, associated)
        success, jpeg = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            raise RuntimeError("Could not encode redetected product crop.")
        return associated, jpeg.tobytes(), inference_ms

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait(timeout=0.25)
                if self._stopping:
                    return
                batch_deadline = time.monotonic() + 0.02
                while (
                    len(self._pending) < self.detector.batch_size
                    and not self._stopping
                ):
                    remaining = batch_deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._condition.wait(timeout=remaining)
                if self._stopping:
                    return
                keys = list(self._pending)[: self.detector.batch_size]
                requests = [self._pending.pop(key) for key in keys]
            started = time.monotonic()
            try:
                with self._inference_lock:
                    detected = self.detector.detect_batch(
                        tuple(request.crop for request in requests)
                    )
            except Exception as exc:
                print(f"PRODUCT_RECOGNITION_ERROR error={exc}")
                continue
            inference_ms = int(round((time.monotonic() - started) * 1000.0))
            completed: list[ProductRecognitionResult] = []
            for request, candidates in zip(requests, detected):
                associated = (
                    tuple(candidates)
                    if request.scope == "full_frame"
                    else tuple(
                        candidate
                        for candidate in candidates
                        if product_center_matches_person(
                            candidate,
                            request.person_box_in_crop,
                            margin_fraction=self.association_margin_fraction,
                        )
                    )
                )
                annotated = request.crop.copy()
                draw_product_detections(annotated, associated)
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not encoded:
                    continue
                try:
                    source_crop_image = encode_lossless_product_crop(request.crop)
                except RuntimeError:
                    continue
                result = ProductRecognitionResult(
                    camera_index=request.camera_index,
                    device_id=request.device_id,
                    track_id=request.track_id,
                    scope=request.scope,
                    rgb_sequence_number=request.rgb_sequence_number,
                    host_synced_seconds=request.host_synced_seconds,
                    observed_at_unix_milliseconds=(
                        request.submitted_at_unix_milliseconds
                    ),
                    inference_milliseconds=inference_ms,
                    crop_box=request.crop_box,
                    person_box_in_crop=request.person_box_in_crop,
                    detections=associated,
                    crop_jpeg=jpeg.tobytes(),
                    source_crop_image=source_crop_image,
                )
                completed.append(result)
                if self.log_results:
                    summary = ",".join(
                        f"{item.label}:{item.score:.3f}" for item in associated
                    ) or "none"
                    print(
                        "PRODUCT_RECOGNITION_TRACE "
                        f"camera_index={request.camera_index} "
                        f"device_id={request.device_id} scope={request.scope} "
                        f"track_id={request.track_id} "
                        f"rgb_sequence={request.rgb_sequence_number} "
                        f"inference_ms={inference_ms} products={summary}"
                    )
            with self._condition:
                self._results.extend(completed)


def product_recognition_payload(
    result: ProductRecognitionResult,
    *,
    visit_id: int | None,
    customer_id: str | None,
    max_age_seconds: float,
) -> dict[str, object]:
    candidates = product_detections_payload(result.detections)
    return {
        "status": "recognized" if candidates else "no_product",
        "freshness": "current",
        "visitId": visit_id,
        "customerId": customer_id,
        "cameraIndex": result.camera_index,
        "deviceId": result.device_id,
        "trackId": result.track_id,
        "scope": result.scope,
        "rgbSequenceNumber": result.rgb_sequence_number,
        "hostSyncedSeconds": result.host_synced_seconds,
        "observedAtUnixMilliseconds": result.observed_at_unix_milliseconds,
        "inferenceMilliseconds": result.inference_milliseconds,
        "maxAgeMilliseconds": int(round(max_age_seconds * 1000.0)),
        "cropBox": {
            "x1": result.crop_box[0],
            "y1": result.crop_box[1],
            "x2": result.crop_box[2],
            "y2": result.crop_box[3],
        },
        "personBoxInCrop": {
            "x1": result.person_box_in_crop[0],
            "y1": result.person_box_in_crop[1],
            "x2": result.person_box_in_crop[2],
            "y2": result.person_box_in_crop[3],
        },
        "bestCandidate": None if not candidates else candidates[0],
        "candidates": candidates,
    }


def product_detections_payload(
    detections: Sequence[ProductDetection],
) -> list[dict[str, object]]:
    return [
        {
            "productId": detection.label.split("_", 1)[0],
            "label": (
                detection.label.split("_", 1)[1]
                if "_" in detection.label
                else detection.label
            ),
            "modelLabel": detection.label,
            "classId": detection.class_id,
            "score": detection.score,
            "boundingBoxInCrop": {
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
            },
        }
        for detection in detections
    ]
