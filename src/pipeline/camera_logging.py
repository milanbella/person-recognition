"""Human-readable camera labels for logs; never change persistent device IDs."""

from collections.abc import Sequence

_camera_numbers: dict[str, int] = {}


def configure_camera_logging(device_ids: Sequence[str]) -> None:
    """Call once before starting workers, using the command-line camera order."""
    global _camera_numbers
    _camera_numbers = {device_id: index for index, device_id in enumerate(device_ids, 1)}


def camera_log_fields(device_id: str) -> str:
    number = _camera_numbers.get(device_id)
    if number is None:
        return f"device_id={device_id}"
    return f"camera_number={number} device_id={device_id}"
