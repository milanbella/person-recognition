import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from pipeline.aruco_markers import ArucoMarkerDetection
from pipeline.depth import CameraIntrinsics
from pipeline.shelf_anchors import (
    ShelfAnchorManager,
    ShelfMarkerDetectionSnapshot,
    configured_shelf_marker_detections,
    sample_shelf_marker_anchor,
)
from pipeline.shelf_config import (
    ShelfDefaults,
    ShelfDefinition,
    ShelfPersonDepthConfig,
    ShelfWatchingConfig,
)
from live_synced_rgbd_streams import load_saved_shelf_anchor_manager
from pipeline.shelf_person_depth import sample_shelf_person_depth
from pipeline.tracking import Track


def _config() -> ShelfWatchingConfig:
    defaults = ShelfDefaults()
    return ShelfWatchingConfig(
        schema_version=1,
        aruco_dictionary="DICT_4X4_50",
        marker_size_mm=80,
        person_depth=ShelfPersonDepthConfig(
            min_valid_pixels=10,
            fallback_center_y_fractions=(0.35,),
        ),
        shelves=(
            ShelfDefinition(
                shelf_id=1,
                label="Shelf",
                marker_id=10,
                approach_distance_mm=defaults.approach_distance_mm,
                departure_distance_mm=defaults.departure_distance_mm,
                approach_dwell_milliseconds=defaults.approach_dwell_milliseconds,
                departure_dwell_milliseconds=defaults.departure_dwell_milliseconds,
                lost_visit_grace_milliseconds=defaults.lost_visit_grace_milliseconds,
                owner_switch_margin_mm=defaults.owner_switch_margin_mm,
                owner_switch_dwell_milliseconds=defaults.owner_switch_dwell_milliseconds,
            ),
        ),
    )


class ShelfDepthAndAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(fx=100, fy=100, cx=50, cy=50)

    def test_torso_sampler_does_not_require_bottom_of_track(self) -> None:
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[38:62, 35:65] = 2000
        track = Track(1, 20, 10, 80, 90, 0.9, status="TRACKED")
        sample = sample_shelf_person_depth(
            depth,
            track,
            intrinsics=self.intrinsics,
            config=_config().person_depth,
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.depth_mm, 2000)
        self.assertLess(sample.anchor_px[1], 70)
        self.assertTrue(np.all(depth[75:90] == 0))

    def test_torso_sampler_uses_configured_central_fallback(self) -> None:
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[28:36, 35:65] = 1750
        track = Track(1, 20, 10, 80, 90, 0.9, status="TRACKED")
        sample = sample_shelf_person_depth(
            depth,
            track,
            intrinsics=self.intrinsics,
            config=_config().person_depth,
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.depth_mm, 1750)
        self.assertLess(sample.anchor_px[1], 50)

    def test_marker_sampler_uses_inset_polygon(self) -> None:
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[40:61, 40:61] = 3000
        detection = ShelfMarkerDetectionSnapshot(
            shelf_id=1,
            marker_id=10,
            corners_px=((40, 40), (60, 40), (60, 60), (40, 60)),
            center_px=(50, 50),
        )
        observation = sample_shelf_marker_anchor(
            depth,
            detection,
            device_id="camera-a",
            intrinsics=self.intrinsics,
            host_synced_seconds=1.0,
            observed_at_unix_milliseconds=1000,
            min_valid_pixels=5,
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.depth_mm, 3000)
        self.assertEqual(observation.point_3d_mm, (0.0, 0.0, 3000.0))

    def test_filters_unconfigured_marker_ids(self) -> None:
        result = configured_shelf_marker_detections(
            (
                ArucoMarkerDetection(10, ((0, 0), (1, 0), (1, 1), (0, 1)), (0.5, 0.5)),
                ArucoMarkerDetection(11, ((0, 0), (1, 0), (1, 1), (0, 1)), (0.5, 0.5)),
            ),
            _config().shelves,
        )
        self.assertEqual([item.marker_id for item in result], [10])

    def test_anchor_manager_auto_calibrates_consistent_observations(self) -> None:
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[40:61, 40:61] = 3000
        detection = ShelfMarkerDetectionSnapshot(
            shelf_id=1,
            marker_id=10,
            corners_px=((40, 40), (60, 40), (60, 60), (40, 60)),
            center_px=(50, 50),
        )
        with tempfile.TemporaryDirectory() as temporary:
            manager = ShelfAnchorManager(
                device_id="camera-a",
                config=_config(),
                calibration_root=Path(temporary),
                min_samples=2,
                max_spread_mm=10,
            )
            for index in range(2):
                manager.process_detections(
                    (detection,),
                    depth_frame_mm=depth,
                    intrinsics=self.intrinsics,
                    host_synced_seconds=float(index),
                    observed_at_unix_milliseconds=1000 + index,
                )
            anchor = manager.anchor_for_shelf(1)
            self.assertIsNotNone(anchor)
            self.assertTrue(manager.calibration_path.exists())
            self.assertEqual(
                [shelf.shelf_id for shelf in manager.anchored_shelves],
                [1],
            )

    def test_live_manager_skips_camera_without_saved_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = load_saved_shelf_anchor_manager(
                device_id="camera-a",
                config=_config(),
                calibration_root=Path(temporary),
            )

        self.assertIsNone(manager)

    def test_live_manager_loads_saved_anchors_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calibration_root = Path(temporary)
            calibration_path = calibration_root / "shelf_anchors_camera-a.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "deviceId": "camera-a",
                        "arucoDictionary": "DICT_4X4_50",
                        "anchors": [
                            {
                                "shelfId": 1,
                                "markerId": 10,
                                "point3dMm": [0, 0, 3000],
                                "sampleCount": 20,
                                "rmsSpreadMm": 2.0,
                                "updatedAtUnixMilliseconds": 1000,
                                "source": "operator_calibrated",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manager = load_saved_shelf_anchor_manager(
                device_id="camera-a",
                config=_config(),
                calibration_root=calibration_root,
            )

        self.assertIsNotNone(manager)
        assert manager is not None
        self.assertEqual([shelf.shelf_id for shelf in manager.anchored_shelves], [1])
        self.assertFalse(manager.auto_save)

    def test_calibration_saves_only_shelves_observed_by_this_camera(self) -> None:
        config = _config()
        second_shelf = replace(
            config.shelves[0],
            shelf_id=2,
            label="Other shelf",
            marker_id=11,
        )
        config = replace(config, shelves=(*config.shelves, second_shelf))
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[40:61, 40:61] = 3000
        first_shelf_detection = ShelfMarkerDetectionSnapshot(
            shelf_id=1,
            marker_id=10,
            corners_px=((40, 40), (60, 40), (60, 60), (40, 60)),
            center_px=(50, 50),
        )

        with tempfile.TemporaryDirectory() as temporary:
            manager = ShelfAnchorManager(
                device_id="camera-a",
                config=config,
                calibration_root=Path(temporary),
                min_samples=2,
                max_spread_mm=10,
                auto_save=False,
                load_existing=False,
            )
            for index in range(2):
                manager.process_detections(
                    (first_shelf_detection,),
                    depth_frame_mm=depth,
                    intrinsics=self.intrinsics,
                    host_synced_seconds=float(index),
                    observed_at_unix_milliseconds=1000 + index,
                )
            manager.accept_candidates()

            payload = json.loads(
                manager.calibration_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                [anchor["shelfId"] for anchor in payload["anchors"]],
                [1],
            )
            self.assertEqual(
                [shelf.shelf_id for shelf in manager.anchored_shelves],
                [1],
            )


if __name__ == "__main__":
    unittest.main()
