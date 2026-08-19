import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

from pipeline.product_training_capture import ProductTrainingCaptureService


class _ControlQueue:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def send(self, message: object) -> None:
        self.messages.append(message)


class _FrameMessage:
    def __init__(self, sequence_number: int) -> None:
        self.sequence_number = sequence_number

    def getCvFrame(self) -> np.ndarray:
        return np.full((8, 12, 3), 127, dtype=np.uint8)

    def getSequenceNum(self) -> int:
        return self.sequence_number

    def getTimestamp(self) -> timedelta:
        return timedelta(seconds=12.5)


class _FrameQueue:
    def __init__(self, frame: _FrameMessage) -> None:
        self.frame = frame
        self.stale = _FrameMessage(1)
        self.timeout: timedelta | None = None

    def tryGet(self):
        stale, self.stale = self.stale, None
        return stale

    def get(self, timeout: timedelta):
        self.timeout = timeout
        return self.frame


class ProductTrainingCaptureTests(unittest.TestCase):
    def test_capture_writes_jpeg_and_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control_queue = _ControlQueue()
            frame_queue = _FrameQueue(_FrameMessage(27))
            service = ProductTrainingCaptureService(
                root=Path(temporary),
                jpeg_quality=93,
                timeout_seconds=2.5,
                target_width=6,
                target_height=4,
            )
            service.register_camera(
                camera_index=1,
                device_id="camera-b",
                control_queue=control_queue,
                frame_queue=frame_queue,
                context_provider=lambda: {"observerSnapshot": {"visitId": 9}},
            )

            result = service.capture(1)

            image_path = Path(result["imagePath"])
            metadata_path = Path(result["metadataPath"])
            self.assertTrue(image_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(image_path.parent.name, "camera-2")
            self.assertEqual(len(control_queue.messages), 1)
            self.assertTrue(control_queue.messages[0].getCaptureStill())
            self.assertEqual(frame_queue.timeout, timedelta(seconds=2.5))

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["cameraIndex"], 1)
            self.assertEqual(metadata["cameraNumber"], 2)
            self.assertEqual(metadata["deviceId"], "camera-b")
            self.assertEqual(metadata["rgbSequenceNumber"], 27)
            self.assertEqual(metadata["width"], 6)
            self.assertEqual(metadata["height"], 4)
            self.assertEqual(metadata["sourceWidth"], 12)
            self.assertEqual(metadata["sourceHeight"], 8)
            self.assertEqual(metadata["resizeMode"], "center_crop")
            self.assertEqual(metadata["jpegQuality"], 93)
            self.assertEqual(metadata["context"]["observerSnapshot"]["visitId"], 9)

    def test_unknown_camera_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = ProductTrainingCaptureService(root=Path(temporary))
            with self.assertRaises(KeyError):
                service.capture(99)


if __name__ == "__main__":
    unittest.main()
