import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from pipeline.detection import (
    DETECTOR_BACKEND_CHOICES,
    YoloOnnxPersonDetector,
    decode_yolo_person_output,
    letterbox_yolo_frame,
    nms_xyxy,
)
from pipeline.config import (
    DEFAULT_DETECTION_INPUT_HEIGHT,
    DEFAULT_DETECTION_INPUT_WIDTH,
    DEFAULT_PERSON_DETECTOR_BACKEND,
    DEFAULT_PERSON_DETECTOR_MODEL,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)


class YoloDetectorTests(unittest.TestCase):
    def test_yolo26n_is_the_only_default_detector(self) -> None:
        self.assertEqual(DETECTOR_BACKEND_CHOICES, ["yolo"])
        self.assertEqual(DEFAULT_PERSON_DETECTOR_BACKEND, "yolo")
        self.assertEqual(DEFAULT_PERSON_DETECTOR_MODEL.name, "yolo26n.onnx")
        self.assertEqual((PREVIEW_WIDTH, PREVIEW_HEIGHT), (3840, 2160))
        self.assertEqual((DEFAULT_DETECTION_INPUT_WIDTH, DEFAULT_DETECTION_INPUT_HEIGHT), (640, 384))

    def test_letterbox_preserves_aspect_ratio(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tensor, scale, padding = letterbox_yolo_frame(frame, (640, 640))

        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(padding, (0.0, 140.0))

    def test_decodes_ultralytics_yolov8_layout_and_person_class(self) -> None:
        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, 0] = [100.0, 120.0, 40.0, 60.0]
        output[0, 4, 0] = 0.9
        output[0, :4, 1] = [200.0, 220.0, 50.0, 70.0]
        output[0, 5, 1] = 0.95

        boxes, scores = decode_yolo_person_output(
            output,
            person_class_id=0,
            score_threshold=0.5,
        )

        np.testing.assert_allclose(boxes, [[80.0, 90.0, 120.0, 150.0]])
        np.testing.assert_allclose(scores, [0.9])

    def test_decodes_yolov5_objectness_layout(self) -> None:
        output = np.zeros((1, 1, 85), dtype=np.float32)
        output[0, 0, :4] = [100.0, 100.0, 20.0, 40.0]
        output[0, 0, 4] = 0.8
        output[0, 0, 5] = 0.75

        boxes, scores = decode_yolo_person_output(
            output,
            person_class_id=0,
            score_threshold=0.5,
        )

        np.testing.assert_allclose(boxes, [[90.0, 80.0, 110.0, 120.0]])
        np.testing.assert_allclose(scores, [0.6])

    def test_decodes_end_to_end_nms_layout(self) -> None:
        output = np.asarray(
            [[[10.0, 20.0, 100.0, 200.0, 0.9, 0.0], [5.0, 5.0, 20.0, 20.0, 0.99, 2.0]]],
            dtype=np.float32,
        )
        boxes, scores = decode_yolo_person_output(
            output,
            person_class_id=0,
            score_threshold=0.5,
        )

        np.testing.assert_allclose(boxes, [[10.0, 20.0, 100.0, 200.0]])
        np.testing.assert_allclose(scores, [0.9])

    def test_nms_keeps_highest_scoring_overlapping_box(self) -> None:
        boxes = np.asarray(
            [[0.0, 0.0, 100.0, 100.0], [5.0, 5.0, 95.0, 95.0], [200.0, 200.0, 250.0, 250.0]],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

        self.assertEqual(nms_xyxy(boxes, scores, 0.5), [0, 2])

    def test_detector_maps_letterboxed_box_back_to_rgb_frame(self) -> None:
        class FakeInput:
            name = "images"
            shape = [1, 3, 640, 640]

        class FakeSession:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def get_inputs(self):
                return [FakeInput()]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _outputs, feed):
                self.last_feed = feed
                output = np.zeros((1, 84, 1), dtype=np.float32)
                output[0, :4, 0] = [320.0, 320.0, 320.0, 320.0]
                output[0, 4, 0] = 0.9
                return [output]

        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "yolo.onnx"
            model_path.touch()
            with (
                patch("pipeline.detection.prepare_onnx_runtime", return_value=["CPUExecutionProvider"]),
                patch("pipeline.detection.ort.InferenceSession", FakeSession),
            ):
                detector = YoloOnnxPersonDetector(
                    model_path=model_path,
                    input_size=(640, 640),
                    score_threshold=0.5,
                    nms_threshold=0.45,
                )
                detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual((detection.x1, detection.y1, detection.x2, detection.y2), (320, 40, 960, 680))
        self.assertAlmostEqual(detection.score, 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
