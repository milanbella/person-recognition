import unittest

import numpy as np

from pipeline.depth import (
    CameraIntrinsics,
    DepthEntranceState,
    Plane3D,
    process_depth_entrance_logic,
    process_depth_plane_logic,
)
from pipeline.tracking import Track


INTRINSICS = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=50.0, cy=50.0)
PLANE = Plane3D(point_mm=(0.0, 0.0, 1000.0), normal=(0.0, 0.0, 1.0))


def depth_frame(depth_mm: int) -> np.ndarray:
    return np.full((100, 100), depth_mm, dtype=np.uint16)


def track(track_id: int, status: str, *, x_offset: int = 0) -> Track:
    return Track(
        track_id=track_id,
        x1=30 + x_offset,
        y1=20,
        x2=70 + x_offset,
        y2=80,
        score=0.9,
        status=status,
    )


def process_plane(
    tracks: list[Track],
    depth_mm: int,
    states: dict[int, DepthEntranceState],
    host_seconds: float,
):
    return process_depth_plane_logic(
        tracks=tracks,
        depth_frame_mm=depth_frame(depth_mm),
        intrinsics=INTRINSICS,
        states=states,
        plane=PLANE,
        plane_enter_direction="positive_to_negative",
        plane_hysteresis_mm=150.0,
        min_valid_pixels=1,
        roi_width_fraction=0.5,
        roi_height_fraction=0.5,
        host_seconds=host_seconds,
        track_split_recovery=True,
        track_split_recovery_max_age_seconds=1.0,
        track_split_recovery_max_centroid_distance_px=220.0,
    )


class DepthCrossingTrackStatusTests(unittest.TestCase):
    def test_lost_track_cannot_generate_direct_plane_leave_from_stale_box(self) -> None:
        states: dict[int, DepthEntranceState] = {}
        process_plane([track(1, "TRACKED")], 1300, states, 1.0)
        entered = process_plane([track(1, "TRACKED")], 800, states, 1.1)

        self.assertEqual(entered.entered_track_ids, [1])
        stale_background = process_plane([track(1, "LOST")], 1400, states, 1.2)

        self.assertEqual(stale_background.exited_track_ids, [])
        self.assertNotIn(1, stale_background.depth_samples)
        self.assertTrue(states[1].entered)
        self.assertEqual(states[1].last_signed_distance_mm, -200.0)

    def test_new_active_track_can_recover_leave_from_lost_entered_track(self) -> None:
        states: dict[int, DepthEntranceState] = {}
        process_plane([track(1, "TRACKED")], 1300, states, 1.0)
        process_plane([track(1, "TRACKED")], 800, states, 1.1)
        process_plane([track(1, "LOST")], 1400, states, 1.2)

        recovered = process_plane(
            [track(1, "LOST"), track(2, "NEW", x_offset=2)],
            1400,
            states,
            1.3,
        )

        self.assertEqual(recovered.exited_track_ids, [2])
        self.assertEqual(recovered.leave_reasons_by_track[2], "track_split_recovery")
        self.assertEqual(recovered.recovered_leave_source_track_ids[2], 1)

    def test_new_active_track_can_recover_entry_without_lost_track_sampling(self) -> None:
        states: dict[int, DepthEntranceState] = {}
        process_plane([track(1, "TRACKED")], 1300, states, 1.0)
        lost = process_plane([track(1, "LOST")], 800, states, 1.1)

        self.assertEqual(lost.entered_track_ids, [])
        recovered = process_plane(
            [track(1, "LOST"), track(2, "NEW", x_offset=2)],
            800,
            states,
            1.2,
        )

        self.assertEqual(recovered.entered_track_ids, [2])
        self.assertEqual(recovered.entry_reasons_by_track[2], "track_split_recovery")
        self.assertEqual(recovered.recovered_entry_source_track_ids[2], 1)

    def test_lost_track_cannot_generate_threshold_crossing(self) -> None:
        states: dict[int, DepthEntranceState] = {}
        common = {
            "intrinsics": INTRINSICS,
            "states": states,
            "depth_threshold_mm": 2000.0,
            "depth_hysteresis_mm": 250.0,
            "min_valid_pixels": 1,
            "roi_width_fraction": 0.5,
            "roi_height_fraction": 0.5,
        }
        process_depth_entrance_logic(
            tracks=[track(1, "TRACKED")],
            depth_frame_mm=depth_frame(2500),
            **common,
        )
        result = process_depth_entrance_logic(
            tracks=[track(1, "LOST")],
            depth_frame_mm=depth_frame(1500),
            **common,
        )

        self.assertEqual(result.entered_track_ids, [])
        self.assertNotIn(1, result.depth_samples)
        self.assertEqual(states[1].last_depth_mm, 2500.0)


if __name__ == "__main__":
    unittest.main()
