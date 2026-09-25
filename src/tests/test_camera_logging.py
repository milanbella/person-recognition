import unittest

from pipeline.camera_logging import camera_log_fields, configure_camera_logging


class CameraLoggingTests(unittest.TestCase):
    def tearDown(self):
        configure_camera_logging([])

    def test_command_line_order_not_device_sort_order(self):
        configure_camera_logging(["zzz", "aaa"])
        self.assertEqual(camera_log_fields("zzz"), "camera_number=1 device_id=zzz")
        self.assertEqual(camera_log_fields("aaa"), "camera_number=2 device_id=aaa")

    def test_unknown_device_is_not_assigned_an_invented_number(self):
        configure_camera_logging(["aaa"])
        self.assertEqual(camera_log_fields("other"), "device_id=other")

    def test_reconfigure_replaces_previous_mapping(self):
        configure_camera_logging(["aaa", "bbb"])
        configure_camera_logging(["bbb"])
        self.assertEqual(camera_log_fields("bbb"), "camera_number=1 device_id=bbb")
        self.assertEqual(camera_log_fields("aaa"), "device_id=aaa")
