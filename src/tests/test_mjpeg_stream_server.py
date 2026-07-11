import threading
import time
import unittest

import numpy as np
from fastapi import HTTPException

from pipeline.mjpeg_stream_server import MjpegStreamServer
from pipeline.observer_api import ObserverCameraSnapshot


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

    def test_observer_snapshot_startup_active_and_stale_states(self) -> None:
        starting = self.server.observer_snapshot_payload(0)
        self.assertEqual(starting["camera"]["status"], "starting")
        self.assertEqual(starting["observations"], [])

        snapshot = ObserverCameraSnapshot(
            camera_index=0,
            device_id="camera-a",
            camera_role="observer",
            rgb_sequence_number=1,
            host_synced_seconds=2.0,
            published_at_unix_milliseconds=3,
            frame_width=8,
            frame_height=8,
            observations=(),
        )
        self.server.publish_observer_snapshot(0, snapshot)
        self.assertEqual(self.server.observer_snapshot_payload(0)["camera"]["status"], "active")
        time.sleep(0.15)
        stale = self.server.observer_snapshot_payload(0)
        self.assertEqual(stale["camera"]["status"], "offline")
        self.assertEqual(stale["observations"], [])

    def test_entrance_only_camera_rejects_observer_snapshots(self) -> None:
        server = MjpegStreamServer(
            camera_device_ids=["entrance-camera"],
            camera_roles=["entrance"],
        )
        try:
            with self.assertRaises(ValueError):
                server.observer_snapshot_payload(0)
        finally:
            server.stop()

    def test_observer_route_returns_404_and_409(self) -> None:
        server = MjpegStreamServer(
            camera_device_ids=["entrance-camera"],
            camera_roles=["entrance"],
        )
        try:
            route = next(
                route
                for route in server.app.routes
                if getattr(route, "path", None) == "/observer-cameras/{cam_index}/observations"
            )
            with self.assertRaises(HTTPException) as unknown:
                route.endpoint(9)
            self.assertEqual(unknown.exception.status_code, 404)

            with self.assertRaises(HTTPException) as wrong_role:
                route.endpoint(0)
            self.assertEqual(wrong_role.exception.status_code, 409)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
