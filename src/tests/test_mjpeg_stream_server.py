import threading
import time
import unittest
from dataclasses import replace

import numpy as np
from fastapi import HTTPException

from pipeline.mjpeg_stream_server import MjpegStreamServer
from pipeline.observer_api import ObserverCameraSnapshot
from pipeline.product_detection import ProductDetection, ProductRecognitionResult
from pipeline.shelf_api import ShelfCameraSnapshot
from pipeline.shelf_config import ShelfDefinition


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
        time.sleep(0.15)
        self.assertEqual(self.server.camera_status_payload()["cameras"][1]["status"], "offline")

    def test_operator_console_routes_and_state_are_disabled_by_default(self) -> None:
        paths = {getattr(route, "path", None) for route in self.server.app.routes}

        self.assertIsNone(self.server.operator_state)
        self.assertIsNone(self.server.operator_store)
        self.assertNotIn("/operator/", paths)
        self.assertFalse(
            any(
                isinstance(path, str) and path.startswith("/operator/api/")
                for path in paths
            )
        )

    def test_stream_returns_latest_jpeg(self) -> None:
        self.server.publish(0, np.zeros((8, 8, 3), dtype=np.uint8))
        part = next(self.server._generate_stream(0))
        self.assertTrue(part.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertIn(b"\xff\xd8", part)

    def test_mjpeg_can_be_disabled_without_disabling_http_api(self) -> None:
        server = MjpegStreamServer(
            camera_device_ids=["camera-a"],
            enable_mjpeg_streaming=False,
        )
        try:
            paths = {getattr(route, "path", None) for route in server.app.routes}
            self.assertIn("/cameras-status", paths)
            stream_route = next(
                route
                for route in server.app.routes
                if getattr(route, "path", None) == "/stream/{cam_index}"
            )
            with self.assertRaises(HTTPException) as disabled:
                stream_route.endpoint(0)
            self.assertEqual(disabled.exception.status_code, 503)
            with self.assertRaisesRegex(RuntimeError, "publication is disabled"):
                server.publish(0, np.zeros((8, 8, 3), dtype=np.uint8))
        finally:
            server.stop()

    def test_stream_waits_for_first_frame(self) -> None:
        generator = self.server._generate_stream(0)
        result: list[bytes] = []
        reader = threading.Thread(target=lambda: result.append(next(generator)))
        reader.start()
        self.server.publish(0, np.zeros((8, 8, 3), dtype=np.uint8))
        reader.join(timeout=1.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(len(result), 1)

    def test_product_recognition_publishes_payload_and_crop(self) -> None:
        result = ProductRecognitionResult(
            camera_index=0,
            device_id="camera-a",
            track_id=2,
            scope="person",
            rgb_sequence_number=10,
            host_synced_seconds=2.0,
            observed_at_unix_milliseconds=time.time_ns() // 1_000_000,
            inference_milliseconds=30,
            crop_box=(1, 2, 100, 200),
            person_box_in_crop=(10, 20, 80, 180),
            detections=(
                ProductDetection(20, 30, 40, 60, 0.9, 1, "001_cola"),
            ),
            crop_jpeg=b"jpeg-data",
        )

        self.server.publish_product_recognition(
            result,
            visit_id=7,
            customer_id=None,
            max_age_seconds=3.0,
        )

        payload = self.server.product_recognition_payload(7)
        self.assertEqual(payload["bestCandidate"]["productId"], "001")
        route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None)
            == "/world-state/visits/{visit_id}/product-crop.jpg"
        )
        response = route.endpoint(7)
        self.assertEqual(response.body, b"jpeg-data")

        observations = self.server.product_camera_observations_payload(7)
        self.assertEqual(len(observations["cameras"]), 2)
        camera_a = observations["cameras"][0]
        self.assertEqual(camera_a["status"], "recognized")
        self.assertEqual(camera_a["trackId"], 2)
        self.assertTrue(camera_a["cropAvailable"])
        self.assertEqual(camera_a["candidates"][0]["label"], "cola")
        camera_b = observations["cameras"][1]
        self.assertEqual(camera_b["status"], "unknown")
        self.assertFalse(camera_b["cropAvailable"])

        observations_route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None)
            == "/world-state/visits/{visit_id}/product-observations"
        )
        self.assertEqual(
            observations_route.endpoint(7)["cameras"][0]["cameraIndex"],
            0,
        )
        crop_route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None)
            == (
                "/world-state/visits/{visit_id}/product-observations/"
                "{camera_index}/crop.jpg"
            )
        )
        self.assertEqual(crop_route.endpoint(7, 0).body, b"jpeg-data")
        with self.assertRaises(HTTPException) as missing_crop:
            crop_route.endpoint(7, 1)
        self.assertEqual(missing_crop.exception.status_code, 404)

        snapshot_route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None)
            == (
                "/world-state/visits/{visit_id}/product-observations/"
                "{camera_index}/snapshot"
            )
        )
        snapshot = snapshot_route.endpoint(7, 0)
        self.assertEqual(snapshot["camera"]["trackId"], 2)
        self.assertEqual(snapshot["requestedVisitId"], 7)

        self.server.publish_product_recognition(
            replace(
                result,
                observed_at_unix_milliseconds=(
                    result.observed_at_unix_milliseconds + 1
                ),
                crop_jpeg=b"new-jpeg-data",
            ),
            visit_id=7,
            customer_id=None,
            max_age_seconds=3.0,
        )
        snapshot_crop_route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None)
            == "/product-observation-snapshots/{snapshot_id}/crop.jpg"
        )
        self.assertEqual(
            snapshot_crop_route.endpoint(snapshot["snapshotId"]).body,
            b"jpeg-data",
        )

    def test_full_frame_product_evidence_is_visible_without_visit_belief(self) -> None:
        observed_ms = time.time_ns() // 1_000_000
        result = ProductRecognitionResult(
            camera_index=1,
            device_id="camera-b",
            track_id=None,
            scope="full_frame",
            rgb_sequence_number=20,
            host_synced_seconds=3.0,
            observed_at_unix_milliseconds=observed_ms,
            inference_milliseconds=40,
            crop_box=(0, 0, 1920, 1080),
            person_box_in_crop=(0, 0, 1920, 1080),
            detections=(
                ProductDetection(100, 200, 300, 500, 0.8, 10, "010_mirinda"),
            ),
            crop_jpeg=b"full-frame-jpeg",
        )

        self.server.publish_camera_product_recognition(
            result,
            max_age_seconds=3.0,
        )

        visit_payload = self.server.product_recognition_payload(7)
        self.assertEqual(visit_payload["status"], "unknown")
        cameras = self.server.product_camera_observations_payload(7)["cameras"]
        self.assertEqual(cameras[1]["scope"], "full_frame")
        self.assertIsNone(cameras[1]["visitId"])
        self.assertEqual(cameras[1]["bestCandidate"]["productId"], "010")

        crop_route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None)
            == (
                "/world-state/visits/{visit_id}/product-observations/"
                "{camera_index}/crop.jpg"
            )
        )
        self.assertEqual(crop_route.endpoint(7, 1).body, b"full-frame-jpeg")

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

    def test_shelf_snapshot_and_event_feed(self) -> None:
        shelf = ShelfDefinition(
            shelf_id=1,
            label="Drinks",
            marker_id=10,
            approach_distance_mm=900,
            departure_distance_mm=1100,
            approach_dwell_milliseconds=500,
            departure_dwell_milliseconds=500,
            lost_visit_grace_milliseconds=1000,
            owner_switch_margin_mm=100,
            owner_switch_dwell_milliseconds=300,
        )
        snapshot = ShelfCameraSnapshot(
            camera_index=0,
            device_id="camera-a",
            camera_role="observer",
            rgb_sequence_number=1,
            depth_sequence_number=1,
            host_synced_seconds=2.0,
            published_at_unix_milliseconds=3,
            shelves=(shelf,),
            anchors_by_shelf={},
            observations=(),
            states_by_shelf={1: "far"},
        )
        self.server.publish_shelf_snapshot(0, snapshot)
        payload = self.server.shelf_snapshot_payload(0)
        self.assertEqual(payload["camera"]["status"], "active")
        self.assertEqual(payload["shelves"][0]["shelfId"], 1)

        self.server.publish_shelf_event_payloads(
            [
                {"eventId": 2, "eventType": "shelf_approach"},
                {"eventId": 3, "eventType": "shelf_departure"},
            ]
        )
        route = next(
            route
            for route in self.server.app.routes
            if getattr(route, "path", None) == "/shelf-events"
        )
        events = route.endpoint(afterEventId=2, limit=100)
        self.assertEqual([event["eventId"] for event in events["events"]], [3])


if __name__ == "__main__":
    unittest.main()
