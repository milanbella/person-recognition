from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo import get_model

from pipeline.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_DETECTION_NMS_THRESHOLD,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_SCRFD_MODEL,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str = "person"


def build_detection_argparser(
    description: str = "Step 2: host-side SCRFD detection on OAK USB frames.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_detection_args(parser)
    return parser


def add_detection_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_SCRFD_MODEL,
        help="Path to the host-side SCRFD ONNX model.",
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
        "--fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera output FPS.",
    )
    return parser


class ScrfdInsightFaceDetector:
    def __init__(
        self,
        model_path: Path,
        input_size: Tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        if not model_path.exists():
            raise FileNotFoundError(f"SCRFD ONNX model not found: {model_path}")

        cache_dir = Path(os.environ["MPLCONFIGDIR"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.providers, self.ctx_id = self._select_runtime()
        self.detector = get_model(str(model_path), providers=self.providers)
        self.detector.prepare(
            ctx_id=self.ctx_id,
            input_size=input_size,
            det_thresh=score_threshold,
            nms_thresh=nms_threshold,
        )

    def _select_runtime(self) -> Tuple[List[str], int]:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            try:
                test_session = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                applied = test_session.get_providers()
                if "CUDAExecutionProvider" in applied:
                    print("Using ONNX Runtime CUDAExecutionProvider for SCRFD.")
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
            except Exception as exc:
                print(f"CUDAExecutionProvider unavailable for SCRFD, falling back to CPU: {exc}")

        print("Using ONNX Runtime CPUExecutionProvider for SCRFD.")
        return ["CPUExecutionProvider"], -1

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections, _kpss = self.detector.detect(frame, input_size=self.input_size)
        result: List[Detection] = []
        for row in detections:
            x1, y1, x2, y2, score = row.tolist()
            result.append(
                Detection(
                    x1=max(0, int(round(x1))),
                    y1=max(0, int(round(y1))),
                    x2=max(0, int(round(x2))),
                    y2=max(0, int(round(y2))),
                    score=float(score),
                )
            )
        return result


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

