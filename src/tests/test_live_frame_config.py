import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from live_synced_rgbd_streams import (
    CAMERA_CONNECT_ATTEMPTS,
    CAMERA_CONNECT_RETRY_DELAY_SECONDS,
    CAMERA_START_DELAY_SECONDS,
    ShopApiClient,
    build_argparser,
    placeholder_frame,
    resolve_live_device,
    validate_operator_console_args,
)


class LiveFrameConfigTests(unittest.TestCase):
    def test_shop_api_bind_visit_reports_bound_customer(self) -> None:
        client = ShopApiClient(
            base_url="https://shop.example",
            api_key="key",
            shop_id=7,
            max_age_seconds=30,
            timeout_seconds=2.0,
        )
        with patch.object(
            client,
            "_post",
            side_effect=[{"customerId": "customer-a"}, {"visitId": 12}],
        ) as post:
            result = client.bind_visit(12)

        self.assertEqual(result.status, "bound")
        self.assertEqual(result.customer_id, "customer-a")
        self.assertEqual(post.call_count, 2)

    def test_shop_api_bind_visit_reports_missing_recent_customer(self) -> None:
        client = ShopApiClient(
            base_url="https://shop.example",
            api_key="key",
            shop_id=7,
            max_age_seconds=30,
            timeout_seconds=2.0,
        )
        with patch.object(client, "_post", return_value=None):
            result = client.bind_visit(12)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "no_recent_unbound_customer")

    def test_shop_api_mark_left_returns_persisted_departure(self) -> None:
        client = ShopApiClient(
            base_url="https://shop.example",
            api_key="key",
            shop_id=7,
            max_age_seconds=30,
            timeout_seconds=2.0,
        )
        expected = {
            "shopId": 7,
            "customerId": "customer-a",
            "visitId": 12,
            "shopLeftAt": "2026-08-12T12:00:00Z",
        }
        with patch.object(client, "_post", return_value=expected) as post:
            response = client.mark_left(12)

        self.assertEqual(response, expected)
        post.assert_called_once_with(
            "/shop-api/shopping-customer/mark-left",
            {"shopId": 7, "visitId": 12},
        )

    def test_shop_api_open_shop_uses_configured_shop(self) -> None:
        client = ShopApiClient(
            base_url="https://shop.example",
            api_key="key",
            shop_id=7,
            max_age_seconds=30,
            timeout_seconds=2.0,
        )
        with patch.object(
            client,
            "_post",
            return_value={"shopId": 7, "customerId": "customer-a"},
        ) as post:
            response = client.open_shop()

        self.assertEqual(response["customerId"], "customer-a")
        post.assert_called_once_with(
            "/shop-api/shopping-customer/open-shop",
            {"shopId": 7},
        )

    def test_camera_startup_safeguard_defaults(self) -> None:
        self.assertEqual(CAMERA_CONNECT_ATTEMPTS, 5)
        self.assertEqual(CAMERA_CONNECT_RETRY_DELAY_SECONDS, 2.0)
        self.assertEqual(CAMERA_START_DELAY_SECONDS, 1.0)

    @patch("live_synced_rgbd_streams.time.sleep")
    @patch("live_synced_rgbd_streams.configure_live_device")
    @patch("live_synced_rgbd_streams.dai.Device")
    @patch("live_synced_rgbd_streams.device_identifier")
    @patch("live_synced_rgbd_streams.list_available_devices")
    def test_camera_connection_reenumerates_before_retry(
        self,
        list_devices: MagicMock,
        identify: MagicMock,
        create_device: MagicMock,
        configure_device: MagicMock,
        sleep: MagicMock,
    ) -> None:
        info = object()
        device = MagicMock()
        list_devices.side_effect = [[], [info]]
        identify.return_value = "camera-a"
        create_device.return_value = device

        result = resolve_live_device("camera-a")

        self.assertIs(result, device)
        self.assertEqual(list_devices.call_count, 2)
        create_device.assert_called_once_with("camera-a")
        configure_device.assert_called_once_with(device)
        sleep.assert_called_once_with(2.0)

    @patch("live_synced_rgbd_streams.time.sleep")
    @patch("live_synced_rgbd_streams.list_available_devices", return_value=[])
    def test_camera_connection_stops_after_five_attempts(
        self,
        list_devices: MagicMock,
        sleep: MagicMock,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "after 5 attempts"):
            resolve_live_device("missing-camera")

        self.assertEqual(list_devices.call_count, 5)
        self.assertEqual(
            sleep.call_args_list,
            [call(2.0), call(2.0), call(2.0), call(2.0)],
        )

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

    def test_product_recognition_is_opt_in_with_crop_defaults(self) -> None:
        args = build_argparser().parse_args([])

        self.assertFalse(args.enable_product_recognition)
        self.assertEqual(args.product_model.name, "best.onnx")
        self.assertEqual(args.product_scan_interval_seconds, 1.0)
        self.assertEqual(args.product_crop_margin, 0.30)
        self.assertFalse(args.product_full_frame)

        debug_args = build_argparser().parse_args(["--product-full-frame"])
        self.assertTrue(debug_args.product_full_frame)

    def test_operator_console_is_disabled_with_separate_run_root(self) -> None:
        args = build_argparser().parse_args([])

        self.assertFalse(args.enable_operator_console)
        self.assertIsNone(args.operator_api_token)
        self.assertEqual(args.operator_runs_root, Path("test-runs"))
        self.assertFalse(args.capture_plane_crossing_evidence)
        self.assertEqual(args.plane_crossing_evidence_frame_count, 5)
        self.assertEqual(args.plane_crossing_evidence_jpeg_quality, 85)

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

    def test_plane_crossing_evidence_requires_operator_console(self) -> None:
        args = build_argparser().parse_args(["--capture-plane-crossing-evidence"])

        with self.assertRaisesRegex(ValueError, "requires --enable-operator-console"):
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
