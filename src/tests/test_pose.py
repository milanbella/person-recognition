import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from pipeline.pose import (
    COCO_POSE_LANDMARK_NAMES,
    PoseDetection,
    PoseLandmark,
    YoloOnnxPoseEstimator,
    decode_yolo_pose_output,
    map_pose_detection_to_source,
)


class PoseTests(unittest.TestCase):
    def test_decodes_raw_ultralytics_pose_output(self) -> None:
        output = np.zeros((1, 56, 2), dtype=np.float32)
        output[0, :5, 0] = [320, 320, 320, 320, 0.9]
        for index in range(17):
            output[0, 5 + index * 3 : 8 + index * 3, 0] = [
                100 + index,
                200 + index,
                0.8,
            ]
        output[0, :5, 1] = [100, 100, 20, 20, 0.1]

        boxes, scores, keypoints = decode_yolo_pose_output(
            output, score_threshold=0.5
        )

        np.testing.assert_array_equal(boxes, [[160, 160, 480, 480]])
        np.testing.assert_allclose(scores, [0.9])
        self.assertEqual(keypoints.shape, (1, 17, 3))
        np.testing.assert_allclose(keypoints[0, 9], [109, 209, 0.8])

    def test_decodes_end_to_end_pose_output(self) -> None:
        output = np.zeros((1, 1, 57), dtype=np.float32)
        output[0, 0, :6] = [10, 20, 110, 220, 0.95, 0]
        output[0, 0, 6:] = np.tile([30, 40, 0.7], 17)

        boxes, scores, keypoints = decode_yolo_pose_output(
            output, score_threshold=0.5
        )

        np.testing.assert_array_equal(boxes, [[10, 20, 110, 220]])
        np.testing.assert_allclose(scores, [0.95])
        np.testing.assert_allclose(keypoints[0, 0], [30, 40, 0.7])

    def test_rejects_incompatible_pose_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "Pose output must contain"):
            decode_yolo_pose_output(
                np.zeros((1, 84, 100), dtype=np.float32), score_threshold=0.5
            )

    def test_maps_crop_landmarks_to_normalized_camera_coordinates(self) -> None:
        landmarks = tuple(
            PoseLandmark(name, 200.0, 300.0, 0.8)
            for name in COCO_POSE_LANDMARK_NAMES
        )
        mapped = map_pose_detection_to_source(
            PoseDetection((10, 20, 390, 580), 0.9, landmarks),
            crop_box=(100, 50, 500, 650),
            source_width=1280,
            source_height=720,
        )

        self.assertEqual(mapped.bounding_box, (110, 70, 490, 630))
        self.assertAlmostEqual(mapped.landmarks[9].x, 300 / 1280)
        self.assertAlmostEqual(mapped.landmarks[9].y, 350 / 720)

    def test_estimator_maps_letterbox_keypoints_back_to_crop(self) -> None:
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

            def run(self, _outputs, _feed):
                output = np.zeros((1, 56, 1), dtype=np.float32)
                output[0, :5, 0] = [320, 320, 320, 320, 0.9]
                output[0, 5:, 0] = np.tile([320, 320, 0.8], 17)
                return [output]

        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "pose.onnx"
            model_path.touch()
            with (
                patch("pipeline.pose.prepare_onnx_runtime", return_value=["CPUExecutionProvider"]),
                patch("pipeline.pose.ort.InferenceSession", FakeSession),
            ):
                estimator = YoloOnnxPoseEstimator(model_path)
                result = estimator.estimate(
                    np.zeros((720, 1280, 3), dtype=np.uint8)
                )

        assert result is not None
        self.assertEqual(result.bounding_box, (320, 40, 960, 680))
        self.assertAlmostEqual(result.landmarks[9].x, 640)
        self.assertAlmostEqual(result.landmarks[9].y, 360)


if __name__ == "__main__":
    unittest.main()
