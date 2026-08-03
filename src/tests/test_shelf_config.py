import json
import tempfile
import unittest
from pathlib import Path

from pipeline.shelf_config import (
    load_shelf_config,
    validate_shelf_config_for_live_cameras,
)


def _payload() -> dict:
    return {
        "schemaVersion": 2,
        "arucoDictionary": "DICT_4X4_50",
        "markerSizeMm": 80,
        "defaults": {
            "approachDistanceMm": 900,
            "departureDistanceMm": 1100,
        },
        "shelves": [
            {
                "shelfId": 1,
                "label": "Drinks",
                "markerId": 10,
            }
        ],
    }


class ShelfConfigTests(unittest.TestCase):
    def _load(self, payload: dict):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shelves.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_shelf_config(path)

    def test_loads_defaults_and_torso_depth_settings(self) -> None:
        config = self._load(_payload())
        self.assertEqual(config.shelves[0].approach_distance_mm, 900)
        self.assertEqual(config.person_depth.center_y_fraction, 0.5)
        self.assertEqual(config.person_depth.width_fraction, 0.3)

    def test_label_is_optional(self) -> None:
        payload = _payload()
        del payload["shelves"][0]["label"]

        config = self._load(payload)

        self.assertEqual(config.shelves[0].label, "Shelf 1")

    def test_loads_multiple_markers_for_one_shelf(self) -> None:
        payload = _payload()
        del payload["shelves"][0]["markerId"]
        payload["shelves"][0]["markerIds"] = [10, 13]

        config = self._load(payload)

        self.assertEqual(config.shelves[0].marker_id, 10)
        self.assertEqual(config.shelves[0].all_marker_ids, (10, 13))

    def test_rejects_both_marker_fields(self) -> None:
        payload = _payload()
        payload["shelves"][0]["markerIds"] = [10, 13]

        with self.assertRaisesRegex(ValueError, "both markerId and markerIds"):
            self._load(payload)

    def test_rejects_marker_reused_by_another_shelf(self) -> None:
        payload = _payload()
        payload["shelves"].append({"shelfId": 2, "markerIds": [11, 10]})

        with self.assertRaisesRegex(ValueError, "Duplicate markerId"):
            self._load(payload)

    def test_rejects_reserved_door_marker(self) -> None:
        payload = _payload()
        payload["shelves"][0]["markerId"] = 3
        with self.assertRaisesRegex(ValueError, "reserved door"):
            self._load(payload)

    def test_rejects_unknown_fields(self) -> None:
        payload = _payload()
        payload["shelves"][0]["typoThreshold"] = 100
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            self._load(payload)

    def test_rejects_invalid_threshold_order(self) -> None:
        payload = _payload()
        payload["shelves"][0]["approachDistanceMm"] = 1200
        with self.assertRaisesRegex(ValueError, "departureDistanceMm"):
            self._load(payload)

    def test_validates_live_camera_and_role(self) -> None:
        config = self._load(_payload())
        with self.assertRaisesRegex(ValueError, "at least one observer-capable"):
            validate_shelf_config_for_live_cameras(
                config,
                camera_device_ids=["camera-a"],
                observer_capable_device_ids=set(),
            )

    def test_loads_legacy_camera_assignments_without_retaining_them(self) -> None:
        payload = _payload()
        payload["schemaVersion"] = 1
        payload["shelves"][0]["cameraDeviceIds"] = ["camera-a"]

        config = self._load(payload)

        self.assertEqual(config.schema_version, 1)
        self.assertFalse(hasattr(config.shelves[0], "camera_device_ids"))

    def test_schema_two_rejects_camera_assignments(self) -> None:
        payload = _payload()
        payload["shelves"][0]["cameraDeviceIds"] = ["camera-a"]

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            self._load(payload)


if __name__ == "__main__":
    unittest.main()
