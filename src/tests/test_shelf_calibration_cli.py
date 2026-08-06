import json
import tempfile
import unittest
from pathlib import Path

from calibrate_shelf_anchors import (
    CALIBRATION_CONTINUE,
    CALIBRATION_QUIT_WITHOUT_SAVE,
    CALIBRATION_SAVE,
    _create_manager,
    build_argparser,
    calibration_action_for_key,
    calibration_source,
)
from pipeline.shelf_config import load_shelf_config


class ShelfCalibrationCliTests(unittest.TestCase):
    def test_calibration_session_does_not_load_existing_anchors(self) -> None:
        config = load_shelf_config(Path("config/shelves.json"))
        shelf = config.shelves[0]
        with tempfile.TemporaryDirectory() as temporary:
            calibration_root = Path(temporary)
            calibration_path = calibration_root / "shelf_anchors_camera-1.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "deviceId": "camera-1",
                        "arucoDictionary": config.aruco_dictionary,
                        "anchors": [
                            {
                                "shelfId": shelf.shelf_id,
                                "markerId": shelf.all_marker_ids[0],
                                "point3dMm": [1, 2, 3000],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = build_argparser().parse_args(
                ["--shelf-calibrations-root", str(calibration_root)]
            )

            manager, _ = _create_manager(
                args,
                device_id="camera-1",
                config=config,
            )

        self.assertEqual(manager.anchors, {})

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
