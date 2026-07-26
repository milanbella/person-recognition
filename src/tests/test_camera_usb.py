import unittest

import depthai as dai

from pipeline.camera import format_usb_connection


class _DeviceInfo:
    name = "3.2.1.1"


class _FakeDevice:
    def __init__(self, speed) -> None:
        self._speed = speed

    def getUsbSpeed(self):
        return self._speed

    def getDeviceInfo(self):
        return _DeviceInfo()


class CameraUsbTests(unittest.TestCase):
    def test_high_speed_is_reported_as_usb2(self) -> None:
        message = format_usb_connection(_FakeDevice(dai.UsbSpeed.HIGH))

        self.assertEqual(
            message,
            "usb_speed=HIGH usb_mode=USB2 nominal_mbps=480 "
            "xlink_path=3.2.1.1",
        )

    def test_super_speed_is_reported_as_usb3(self) -> None:
        message = format_usb_connection(_FakeDevice(dai.UsbSpeed.SUPER))

        self.assertIn("usb_speed=SUPER usb_mode=USB3 nominal_mbps=5000", message)

    def test_missing_usb_api_reports_unknown(self) -> None:
        message = format_usb_connection(object())

        self.assertIn("usb_speed=UNKNOWN usb_mode=UNKNOWN", message)
        self.assertIn("xlink_path=unknown", message)


if __name__ == "__main__":
    unittest.main()
