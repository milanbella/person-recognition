import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".cache" / "matplotlib"),
)
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import cv2
import depthai as dai
import numpy as np
import onnxruntime as ort
from insightface.model_zoo import get_model


PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str = "person"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2: host-side SCRFD detection on OAK USB frames."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "scrfd_person_2.5g.onnx",
        help="Path to the host-side SCRFD ONNX model.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=640,
        help="Detector input width.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=640,
        help="Detector input height.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
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


def main() -> None:
    args = build_argparser().parse_args()

    detector = ScrfdInsightFaceDetector(
        model_path=args.model,
        input_size=(args.input_width, args.input_height),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )

    device = dai.Device()
    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()} Platform: {platform}")

    with dai.Pipeline(device) as pipeline:
        print("Step 2: host-side SCRFD detection on OAK USB frames.")

        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )
        queue = camera_out.createOutputQueue(maxSize=4, blocking=False)

        print("Pipeline created. Starting...")
        pipeline.start()

        while pipeline.isRunning():
            msg = queue.get()
            frame = msg.getCvFrame()

            detections = detector.detect(frame)
            draw_detections(frame, detections)

            cv2.imshow("OAK Host SCRFD Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
