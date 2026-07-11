import threading
import time
import unittest

import numpy as np

from pipeline.mjpeg_stream_server import MjpegStreamServer


class MjpegStreamServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MjpegStreamServer(
            camera_device_ids=["camera-a", "camera-b"],
            camera_timeout_seconds=0.05,
        )

    def tearDown(self) -> None:
        self.server.stop()

    def test_status_preserves_order_and_tracks_health(self) -> None:
        cameras = self.server.camera_status_payload()["cameras"]
        self.assertEqual([camera["deviceId"] for camera in cameras], ["camera-a", "camera-b"])
        self.assertEqual([camera["status"] for camera in cameras], ["offline", "offline"])

        self.server.publish(1, np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(self.server.camera_status_payload()["cameras"][1]["status"], "active")
        time.sleep(0.06)
        self.assertEqual(self.server.camera_status_payload()["cameras"][1]["status"], "offline")

    def test_stream_returns_latest_jpeg(self) -> None:
        self.server.publish(0, np.zeros((8, 8, 3), dtype=np.uint8))
        part = next(self.server._generate_stream(0))
        self.assertTrue(part.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertIn(b"\xff\xd8", part)

    def test_stream_waits_for_first_frame(self) -> None:
        generator = self.server._generate_stream(0)
        result: list[bytes] = []
        reader = threading.Thread(target=lambda: result.append(next(generator)))
        reader.start()
        self.server.publish(0, np.zeros((8, 8, 3), dtype=np.uint8))
        reader.join(timeout=1.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(len(result), 1)

    def test_rejects_unknown_camera_index(self) -> None:
        with self.assertRaises(KeyError):
            self.server.publish(9, np.zeros((8, 8, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
