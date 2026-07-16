import unittest

from live_synced_rgbd_streams import build_argparser, placeholder_frame


class LiveFrameConfigTests(unittest.TestCase):
    def test_frame_dimensions_default_to_4k(self) -> None:
        args = build_argparser().parse_args([])

        self.assertEqual((args.frame_width, args.frame_height), (3840, 2160))

    def test_frame_dimensions_can_be_overridden_together(self) -> None:
        args = build_argparser().parse_args(["--frame-width", "1920", "--frame-height", "1080"])
        frame = placeholder_frame("waiting", width=args.frame_width, height=args.frame_height)

        self.assertEqual(frame.shape, (1080, 1920, 3))

    def test_performance_logging_is_opt_in(self) -> None:
        args = build_argparser().parse_args([])

        self.assertFalse(args.log_performance)
        self.assertEqual(args.performance_log_interval_seconds, 5.0)


if __name__ == "__main__":
    unittest.main()
