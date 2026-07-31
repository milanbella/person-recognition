import unittest
from pathlib import Path

from live_synced_rgbd_streams import (
    build_argparser,
    placeholder_frame,
    validate_operator_console_args,
)


class LiveFrameConfigTests(unittest.TestCase):
    def test_live_frame_dimensions_use_balanced_defaults(self) -> None:
        args = build_argparser().parse_args([])

        self.assertEqual((args.frame_width, args.frame_height), (1920, 1080))
        self.assertEqual((args.processing_width, args.processing_height), (1280, 720))
        self.assertEqual(args.max_rgb_depth_delta_ms, 250.0)
        self.assertEqual(args.processing_buffer_seconds, 6.0)
        self.assertEqual(args.depth_median_filter, "7x7")
        self.assertTrue(args.show_annotated_preview)

    def test_frame_dimensions_can_be_overridden_together(self) -> None:
        args = build_argparser().parse_args(["--frame-width", "1920", "--frame-height", "1080"])
        frame = placeholder_frame("waiting", width=args.frame_width, height=args.frame_height)

        self.assertEqual(frame.shape, (1080, 1920, 3))

    def test_performance_logging_is_opt_in(self) -> None:
        args = build_argparser().parse_args([])

        self.assertFalse(args.log_performance)
        self.assertEqual(args.performance_log_interval_seconds, 5.0)

    def test_depth_median_filter_can_be_disabled(self) -> None:
        args = build_argparser().parse_args(["--depth-median-filter", "off"])

        self.assertEqual(args.depth_median_filter, "off")

    def test_annotated_preview_is_default_and_raw_can_be_requested(self) -> None:
        default_args = build_argparser().parse_args([])
        raw_args = build_argparser().parse_args(["--show-raw-preview"])

        self.assertTrue(default_args.show_annotated_preview)
        self.assertFalse(raw_args.show_annotated_preview)

    def test_shelf_watching_is_opt_in(self) -> None:
        args = build_argparser().parse_args([])

        self.assertFalse(args.enable_shelf_watching)
        self.assertEqual(args.shelf_config, Path("config") / "shelves.json")
        self.assertFalse(hasattr(args, "shelf_marker_scan_interval_seconds"))

    def test_operator_console_is_disabled_with_separate_run_root(self) -> None:
        args = build_argparser().parse_args([])

        self.assertFalse(args.enable_operator_console)
        self.assertIsNone(args.operator_api_token)
        self.assertEqual(args.operator_runs_root, Path("test-runs"))

    def test_operator_console_can_be_enabled_with_cli_token(self) -> None:
        args = build_argparser().parse_args(
            [
                "--enable-operator-console",
                "--operator-api-token",
                "test-secret",
            ]
        )

        self.assertTrue(args.enable_operator_console)
        self.assertEqual(args.operator_api_token, "test-secret")
        validate_operator_console_args(args)

    def test_operator_console_requires_enable_flag_and_token_together(self) -> None:
        parser = build_argparser()
        missing_token = parser.parse_args(["--enable-operator-console"])
        token_without_console = parser.parse_args(
            ["--operator-api-token", "test-secret"]
        )
        disabled_streaming = parser.parse_args(
            [
                "--enable-operator-console",
                "--operator-api-token",
                "test-secret",
                "--disable-streaming",
            ]
        )

        with self.assertRaisesRegex(ValueError, "operator-api-token is required"):
            validate_operator_console_args(missing_token)
        with self.assertRaisesRegex(ValueError, "requires --enable"):
            validate_operator_console_args(token_without_console)
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            validate_operator_console_args(disabled_streaming)


if __name__ == "__main__":
    unittest.main()
