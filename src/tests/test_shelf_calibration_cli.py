import unittest
from pathlib import Path

from calibrate_shelf_anchors import (
    CALIBRATION_CONTINUE,
    CALIBRATION_QUIT_WITHOUT_SAVE,
    CALIBRATION_SAVE,
    build_argparser,
    calibration_action_for_key,
    calibration_source,
)


class ShelfCalibrationCliTests(unittest.TestCase):
    def test_device_id_uses_live_camera_by_default(self) -> None:
        args = build_argparser().parse_args(["--device-id", "camera-1"])

        self.assertEqual(calibration_source(args), "live")
        self.assertEqual((args.width, args.height), (1280, 720))
        self.assertEqual(args.depth_median_filter, "7x7")
        self.assertFalse(args.show_rejected_candidates)

    def test_recording_source_can_resolve_recording_by_device_id(self) -> None:
        args = build_argparser().parse_args(
            ["--source", "recording", "--device-id", "camera-1"]
        )

        self.assertEqual(calibration_source(args), "recording")

    def test_explicit_recording_directory_selects_recording_source(self) -> None:
        args = build_argparser().parse_args(
            ["--recording-dir", "recordings/example.rgbd"]
        )

        self.assertEqual(calibration_source(args), "recording")
        self.assertEqual(args.recording_dir, Path("recordings/example.rgbd"))

    def test_q_quits_without_saving(self) -> None:
        self.assertEqual(
            calibration_action_for_key(ord("q"), valid_anchor_count=2),
            CALIBRATION_QUIT_WITHOUT_SAVE,
        )

    def test_s_saves_only_when_an_anchor_is_ready(self) -> None:
        self.assertEqual(
            calibration_action_for_key(ord("s"), valid_anchor_count=1),
            CALIBRATION_SAVE,
        )
        self.assertEqual(
            calibration_action_for_key(ord("s"), valid_anchor_count=0),
            CALIBRATION_CONTINUE,
        )


if __name__ == "__main__":
    unittest.main()
