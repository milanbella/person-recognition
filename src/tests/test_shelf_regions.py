import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pipeline.shelf_regions import (
    CameraShelfRegions,
    NormalizedPoint,
    ShelfDepthCell,
    ShelfDepthModel,
    ShelfRegion,
    build_shelf_depth_model,
    box_center_in_shelf_region,
    load_camera_shelf_regions,
    normalized_point_in_polygon,
    save_camera_shelf_regions,
)


class ShelfRegionTests(unittest.TestCase):
    @staticmethod
    def _depth_model() -> ShelfDepthModel:
        return ShelfDepthModel(
            columns=1,
            rows=1,
            aligned_depth_width=8,
            aligned_depth_height=8,
            minimum_valid_fraction=0.35,
            cells=(ShelfDepthCell(0, 0, 4000.0, 10.0, 64, 1.0),),
            calibration_frame_count=4,
            calibrated_at_unix_milliseconds=123,
            intrinsics_fx=8.0,
            intrinsics_fy=8.0,
            intrinsics_cx=4.0,
            intrinsics_cy=4.0,
            camera_calibration_id="camera-a-calibration",
        )

    def test_round_trip_and_atomic_replacement(self) -> None:
        config = CameraShelfRegions(
            device_id="camera-a",
            regions=(
                ShelfRegion(
                    3,
                    (
                        NormalizedPoint(0.1, 0.2),
                        NormalizedPoint(0.5, 0.2),
                        NormalizedPoint(0.5, 0.8),
                        NormalizedPoint(0.1, 0.8),
                    ),
                    450.0,
                    self._depth_model(),
                    additional_polygons=((
                        NormalizedPoint(0.5, 0.3),
                        NormalizedPoint(0.7, 0.3),
                        NormalizedPoint(0.7, 0.7),
                        NormalizedPoint(0.5, 0.7),
                    ),),
                ),
            ),
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "regions.json"
            save_camera_shelf_regions(path, config)
            restored = load_camera_shelf_regions(path)
            leftovers = list(path.parent.glob("*.tmp"))

        self.assertEqual(restored, config)
        self.assertEqual(leftovers, [])

    def test_builds_robust_depth_grid_and_rejects_invalid_pixels(self) -> None:
        polygon = (
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(1.0, 0.0),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        )
        frames = [np.full((8, 8), 4000, dtype=np.uint16) for _ in range(4)]
        frames[0][0, 0] = 0
        frames[1][0, 0] = 12000
        model = build_shelf_depth_model(
            frames,
            polygons=(polygon,),
            columns=2,
            rows=2,
            minimum_valid_fraction=0.5,
            minimum_valid_samples=4,
            minimum_depth_mm=200,
            maximum_depth_mm=10000,
            calibrated_at_unix_milliseconds=123,
            intrinsics=(8.0, 8.0, 4.0, 4.0),
            camera_calibration_id="camera-a-calibration",
        )

        self.assertEqual(len(model.cells), 4)
        self.assertTrue(all(cell.median_depth_mm == 4000.0 for cell in model.cells))

    def test_rejects_duplicate_shelf_ids(self) -> None:
        payload = {
            "schemaVersion": 1,
            "deviceId": "camera-a",
            "shelves": [
                {
                    "shelfId": 1,
                    "polygonNormalized": [
                        {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}
                    ],
                },
                {
                    "shelfId": 1,
                    "polygonNormalized": [
                        {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}
                    ],
                },
            ],
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "regions.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate shelfId"):
                load_camera_shelf_regions(path)

    def test_point_and_box_membership(self) -> None:
        region = ShelfRegion(
            1,
            (
                NormalizedPoint(0.2, 0.2),
                NormalizedPoint(0.8, 0.2),
                NormalizedPoint(0.8, 0.8),
                NormalizedPoint(0.2, 0.8),
            ),
        )
        self.assertTrue(normalized_point_in_polygon(0.5, 0.5, region.polygon))
        self.assertFalse(normalized_point_in_polygon(0.1, 0.5, region.polygon))
        self.assertTrue(
            box_center_in_shelf_region(
                (400, 300, 600, 500),
                frame_width=1000,
                frame_height=800,
                region=region,
            )
        )

    def test_shelf_region_membership_is_union_of_surface_polygons(self) -> None:
        region = ShelfRegion(
            shelf_id=1,
            polygon=(
                NormalizedPoint(0.1, 0.2),
                NormalizedPoint(0.3, 0.2),
                NormalizedPoint(0.3, 0.8),
                NormalizedPoint(0.1, 0.8),
            ),
            additional_polygons=((
                NormalizedPoint(0.7, 0.2),
                NormalizedPoint(0.9, 0.2),
                NormalizedPoint(0.9, 0.8),
                NormalizedPoint(0.7, 0.8),
            ),),
        )

        self.assertTrue(
            box_center_in_shelf_region(
                (750, 300, 850, 500),
                frame_width=1000,
                frame_height=800,
                region=region,
            )
        )
        self.assertFalse(
            box_center_in_shelf_region(
                (450, 300, 550, 500),
                frame_width=1000,
                frame_height=800,
                region=region,
            )
        )


if __name__ == "__main__":
    unittest.main()
