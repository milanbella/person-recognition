import io
import unittest
from contextlib import redirect_stdout

from pipeline.performance import LivePerformanceLogger


class LivePerformanceLoggerTests(unittest.TestCase):
    def test_disabled_logger_produces_no_output(self) -> None:
        logger = LivePerformanceLogger(enabled=False)
        output = io.StringIO()

        with redirect_stdout(output):
            logger.report(5.0)

        self.assertEqual(output.getvalue(), "")

    def test_report_aggregates_counts_and_stage_averages(self) -> None:
        logger = LivePerformanceLogger(
            enabled=True,
            interval_seconds=5.0,
            window_started=100.0,
        )
        logger.stage_seconds = {
            "yolo": 0.120,
            "face": 0.080,
            "cycle": 0.500,
        }
        logger.stage_counts = {
            "yolo": 4,
            "face": 2,
            "cycle": 5,
        }
        logger.cycle_count = 5
        logger.camera_poll_count = 25
        logger.rgb_frame_count = 20
        logger.processed_frame_count = 16
        logger.stream_frame_count = 10
        logger.detection_count = 24
        logger.track_count = 32
        output = io.StringIO()

        with redirect_stdout(output):
            logger.report(105.0)

        line = output.getvalue()
        self.assertIn("LIVE_PERF window_s=5.0 cycles=5 camera_polls=25", line)
        self.assertIn("rgb_frames=20 processed_frames=16 processed_fps=3.2", line)
        self.assertIn("stream_frames=10 avg_detections=1.50 avg_tracks=2.00", line)
        self.assertIn("yolo_ms=30.0", line)
        self.assertIn("face_ms=40.0", line)
        self.assertIn("cycle_ms=100.0", line)
        self.assertEqual(logger.processed_frame_count, 0)
        self.assertEqual(logger.stage_seconds, {})

    def test_non_positive_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LivePerformanceLogger(enabled=True, interval_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
