from __future__ import annotations

import argparse
from typing import List

import depthai as dai


def add_device_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="Optional OAK device id/MXID to connect to explicitly.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available OAK devices and exit.",
    )
    return parser


def list_available_devices() -> List[dai.DeviceInfo]:
    return list(dai.Device.getAllAvailableDevices())


def device_identifier(info: dai.DeviceInfo) -> str:
    try:
        return str(info.getDeviceId())
    except Exception:
        value = getattr(info, "deviceId", None)
        return "unknown" if value is None else str(value)


def format_device_info(info: dai.DeviceInfo) -> str:
    device_id = device_identifier(info)
    name = getattr(info, "name", None)
    protocol = getattr(info, "protocol", None)
    platform = getattr(info, "platform", None)
    parts = [f"id={device_id}"]
    if name:
        parts.append(f"name={name}")
    if platform is not None:
        parts.append(f"platform={platform}")
    if protocol is not None:
        parts.append(f"protocol={protocol}")
    return " ".join(parts)


def print_available_devices() -> None:
    devices = list_available_devices()
    if not devices:
        print("No OAK devices found.")
        return

    print("Available OAK devices:")
    for info in devices:
        print(f"  {format_device_info(info)}")


def resolve_device(device_id: str | None) -> dai.Device:
    if device_id is None:
        return dai.Device()

    available = list_available_devices()
    matching = [info for info in available if device_identifier(info) == device_id]
    if not matching:
        available_ids = ", ".join(device_identifier(info) for info in available) or "none"
        raise RuntimeError(
            f"Requested device-id '{device_id}' not found. Available device ids: {available_ids}"
        )
    return dai.Device(device_id)


def open_or_list_devices(args: argparse.Namespace) -> dai.Device | None:
    if getattr(args, "list_devices", False):
        print_available_devices()
        return None
    return resolve_device(getattr(args, "device_id", None))


def print_connected_device(device: dai.Device) -> None:
    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()} Platform: {platform}")

