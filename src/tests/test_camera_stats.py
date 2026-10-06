import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from pipeline.camera_stats import CameraStats
from pipeline.mjpeg_stream_server import MjpegStreamServer


class CameraStatsTests(unittest.TestCase):
    def test_camera_status_uses_receipt_without_mjpeg(self):
        server = MjpegStreamServer(
            camera_device_ids=["a", "b"], enable_mjpeg_streaming=False,
            camera_timeout_seconds=3,
        )
        try:
            with patch("pipeline.camera_stats.time.monotonic", return_value=100):
                server.camera_stats[1].record_received_frame(sequence=1, exposure_us=None)
            with patch("pipeline.mjpeg_stream_server.time.monotonic", return_value=102):
                cameras = server.camera_status_payload()["cameras"]
                self.assertEqual([c["status"] for c in cameras], ["offline", "active"])
                self.assertEqual([c["id"] for c in cameras], [0, 1])
            # Re-publishing old images must not hide a stopped raw stream.
            server._cameras[1].last_frame_monotonic = 104
            with patch("pipeline.mjpeg_stream_server.time.monotonic", return_value=104):
                self.assertEqual(server.camera_status_payload()["cameras"][1]["status"], "offline")
        finally:
            server.stop()

    def test_metrics_and_freshness(self):
        stats = CameraStats("oak")
        with patch("pipeline.camera_stats.time.monotonic", return_value=100):
            self.assertEqual(stats.payload(0)["status"], "offline")
            stats.record_telemetry(SimpleNamespace(
                chipTemperature=SimpleNamespace(average=51.25),
                leonCssCpuUsage=SimpleNamespace(average=0.23),
                ddrMemoryUsage=SimpleNamespace(used=25, total=100),
            ))
            stats.record_received_frame(sequence=10, exposure_us=1500)
            stats.record_inference(detections=2, inference_ms=12)
        with patch("pipeline.camera_stats.time.monotonic", return_value=100.5):
            stats.record_received_frame(sequence=13, exposure_us=1400)
            stats.record_received_frame(sequence=13, exposure_us=1400)
            stats.record_inference(detections=0, inference_ms=11)
            stats.set_processing_fps(2)
            data = stats.payload(0)
            self.assertEqual(data["status"], "active")
            self.assertEqual(data["frameAgeMs"], 0)
            self.assertEqual(data["frameCount"], 2)
            self.assertEqual(data["droppedFrames"], 2)
            self.assertEqual(data["sequenceGaps"], 1)
            self.assertEqual(data["droppedFramesTotal"], 2)
            self.assertEqual(data["sequenceGapsTotal"], 1)
            self.assertIsNotNone(datetime.fromisoformat(data["telemetryStartedAt"]).tzinfo)
            self.assertEqual(data["processingFps"], 2)
            self.assertEqual(data["cpuLoad"], 23)
            self.assertEqual(data["memoryLoad"], 25)
            self.assertEqual(data["inferenceMs"], 11)
            self.assertEqual(data["detections"], 0)
        with patch("pipeline.camera_stats.time.monotonic", return_value=102):
            self.assertEqual(stats.payload(0)["status"], "stale")
        with patch("pipeline.camera_stats.time.monotonic", return_value=111):
            data = stats.payload(0)
            self.assertEqual(data["status"], "offline")
            self.assertIsNone(data["temperature"])
            self.assertEqual(data["processingFps"], 2)

    def test_empty_ddr_and_sequence_reset(self):
        stats = CameraStats("oak")
        stats.record_telemetry(SimpleNamespace(
            chipTemperature=SimpleNamespace(average=40),
            leonCssCpuUsage=SimpleNamespace(average=0),
            ddrMemoryUsage=SimpleNamespace(used=0, total=0),
        ))
        for seq in (100, 1):
            stats.record_received_frame(sequence=seq, exposure_us=None)
        data = stats.payload(0)
        self.assertIsNone(data["memoryLoad"])
        self.assertEqual(data["droppedFrames"], 0)

    def test_minute_counters_reset_on_next_received_frame_but_totals_continue(self):
        first = datetime(2026, 10, 3, 16, 35, 59, tzinfo=timezone.utc)
        second = datetime(2026, 10, 3, 16, 36, 0, tzinfo=timezone.utc)
        with patch("pipeline.camera_stats._utc_now", return_value=first):
            stats = CameraStats("oak")
            stats.record_received_frame(sequence=10, exposure_us=None)
            stats.record_received_frame(sequence=13, exposure_us=None)
        with patch("pipeline.camera_stats._utc_now", return_value=second):
            # Checkout resets on the first packet of a new UTC minute, not on a read.
            self.assertEqual(stats.payload(0)["droppedFrames"], 2)
            stats.record_received_frame(sequence=13, exposure_us=None)
            self.assertEqual(stats.payload(0)["droppedFrames"], 0)
            stats.record_received_frame(sequence=16, exposure_us=None)
            data = stats.payload(0)
        self.assertEqual((data["droppedFrames"], data["sequenceGaps"]), (2, 1))
        self.assertEqual((data["droppedFramesTotal"], data["sequenceGapsTotal"]), (4, 2))

    def test_restart_gives_fresh_totals_and_a_shared_start_timestamp(self):
        first = CameraStats("first")
        first.record_received_frame(sequence=1, exposure_us=None)
        first.record_received_frame(sequence=4, exposure_us=None)
        second = CameraStats("second")
        self.assertEqual(second.payload(1)["droppedFramesTotal"], 0)
        self.assertEqual(first.payload(0)["telemetryStartedAt"], second.payload(1)["telemetryStartedAt"])

    def test_received_packets_count_even_without_inference(self):
        stats = CameraStats("oak")
        for seq in range(20):
            stats.record_rgb_packet(SimpleNamespace(getSequenceNum=lambda: seq))
        self.assertEqual(stats.payload(0)["frameCount"], 20)
        self.assertEqual(stats.payload(0)["droppedFrames"], 0)
        self.assertIsNone(stats.payload(0)["lastInferenceTimestamp"])
        stats.record_inference(detections=1, inference_ms=9)
        self.assertEqual(stats.payload(0)["frameCount"], 20)

    def test_shared_loop_fps_averaged_over_window(self):
        server = MjpegStreamServer(camera_device_ids=["a", "b"])
        try:
            with patch("pipeline.mjpeg_stream_server.time.monotonic", side_effect=[10, 10.5, 11.5]):
                for _ in range(3):
                    server.record_processing_cycle()
            for stats in server.camera_stats.values():
                self.assertEqual(stats.payload(0)["processingFps"], 2)
        finally:
            server.stop()

    def test_stream_contract_with_mjpeg_disabled(self):
        async def check():
            server = MjpegStreamServer(camera_device_ids=["oak"], enable_mjpeg_streaming=False)
            try:
                route = next(r for r in server.app.routes if r.path == "/stats/{cam_index}")
                request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
                for index, status in ((-1, 400), (1, 404), (10, 404)):
                    with self.assertRaises(HTTPException) as raised:
                        await route.endpoint(index, request)
                    self.assertEqual(raised.exception.status_code, status)
                response = await route.endpoint(0, request)
                self.assertEqual(response.media_type, "application/x-ndjson")
                self.assertEqual(response.headers["x-accel-buffering"], "no")
                line = await anext(response.body_iterator)
                self.assertTrue(line.endswith("\n"))
                payload = json.loads(line)
                self.assertEqual(payload["camera"], 0)
                self.assertEqual(payload["deviceId"], "oak")
                self.assertEqual(payload["status"], "offline")
                with patch("pipeline.mjpeg_stream_server.asyncio.sleep", new_callable=AsyncMock) as sleep:
                    next_line = await anext(response.body_iterator)
                    sleep.assert_awaited_once_with(2)
                    self.assertEqual(json.loads(next_line)["deviceId"], "oak")
                self.assertEqual(set(payload), {
                    "camera", "timestamp", "temperature", "cpuLoad", "memoryLoad",
                    "status", "checkoutActive", "deviceId", "frameAgeMs", "frameCount",
                    "sequenceNumber", "droppedFrames", "sequenceGaps",
                    "droppedFramesTotal", "sequenceGapsTotal", "telemetryStartedAt", "exposureUs",
                    "detections", "processingFps", "inferenceMs", "lastFrameTimestamp",
                    "lastDetectionTimestamp", "lastInferenceTimestamp",
                })
                await response.body_iterator.aclose()
                request.is_disconnected.return_value = True
                response = await route.endpoint(0, request)
                with self.assertRaises(StopAsyncIteration):
                    await anext(response.body_iterator)
                request.is_disconnected.return_value = False
                server._stopping = True
                response = await route.endpoint(0, request)
                with self.assertRaises(StopAsyncIteration):
                    await anext(response.body_iterator)
            finally:
                server.stop()
        asyncio.run(check())
