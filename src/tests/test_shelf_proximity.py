import unittest
from dataclasses import replace

from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_config import (
    ShelfDefinition,
    ShelfPersonDepthConfig,
    ShelfWatchingConfig,
)
from pipeline.shelf_proximity import (
    ShelfCameraObservation,
    ShelfProximityCoordinator,
    person_to_shelf_distance_mm,
)


def _shelf(
    shelf_id: int = 1,
    marker_id: int = 10,
    label: str = "Drinks",
) -> ShelfDefinition:
    return ShelfDefinition(
        shelf_id=shelf_id,
        label=label,
        marker_id=marker_id,
        approach_distance_mm=900,
        departure_distance_mm=1100,
        approach_dwell_milliseconds=500,
        departure_dwell_milliseconds=500,
        lost_visit_grace_milliseconds=1000,
        owner_switch_margin_mm=100,
        owner_switch_dwell_milliseconds=300,
    )


def _config() -> ShelfWatchingConfig:
    return ShelfWatchingConfig(
        schema_version=1,
        aruco_dictionary="DICT_4X4_50",
        marker_size_mm=80,
        person_depth=ShelfPersonDepthConfig(),
        shelves=(_shelf(),),
    )


def _observation(
    *,
    visit_id: int,
    track_id: int,
    distance_mm: float,
    now_ms: int,
    camera_index: int = 0,
    marker_id: int = 10,
    shelf_id: int = 1,
    shelf_label: str = "Drinks",
) -> ShelfCameraObservation:
    device_id = "camera-a" if camera_index == 0 else "camera-b"
    anchor = ShelfAnchor(
        shelf_id=shelf_id,
        marker_id=marker_id,
        device_id=device_id,
        point_3d_mm=(0, 0, 3000),
        sample_count=20,
        rms_spread_mm=5,
        updated_at_unix_milliseconds=now_ms,
        source="operator_calibrated",
    )
    return ShelfCameraObservation(
        shelf_id=shelf_id,
        shelf_label=shelf_label,
        marker_id=marker_id,
        camera_index=camera_index,
        device_id=device_id,
        track_id=track_id,
        visit_id=visit_id,
        visit_origin="entrance_confirmed",
        customer_id=f"customer-{visit_id}",
        distance_mm=distance_mm,
        person_point_3d_mm=(0, 0, 3000 - distance_mm),
        anchor=anchor,
        host_synced_seconds=now_ms / 1000,
        observed_at_unix_milliseconds=now_ms,
        rgb_sequence_number=now_ms,
        depth_sequence_number=now_ms,
    )


class ShelfProximityTests(unittest.TestCase):
    def test_distance_is_euclidean_3d(self) -> None:
        self.assertEqual(
            person_to_shelf_distance_mm((3, 4, 12), (0, 0, 0)),
            13,
        )

    def test_approach_event_uses_nearest_observation_marker(self) -> None:
        shelf = replace(_shelf(), marker_ids=(10, 13))
        coordinator = ShelfProximityCoordinator(replace(_config(), shelves=(shelf,)))
        coordinator.update_camera(
            camera_index=0,
            observations=(
                _observation(
                    visit_id=4,
                    track_id=7,
                    marker_id=13,
                    distance_mm=800,
                    now_ms=1000,
                ),
            ),
            host_synced_seconds=1,
            now_unix_milliseconds=1000,
        )
        events = coordinator.update_camera(
            camera_index=0,
            observations=(
                _observation(
                    visit_id=4,
                    track_id=7,
                    marker_id=13,
                    distance_mm=790,
                    now_ms=1500,
                ),
            ),
            host_synced_seconds=1.5,
            now_unix_milliseconds=1500,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].shelf_id, 1)
        self.assertEqual(events[0].marker_id, 13)

    def test_restore_ignores_session_with_stale_marker_mapping(self) -> None:
        coordinator = ShelfProximityCoordinator(_config())
        coordinator.restore_near_session(
            shelf_id=1,
            visit_id=4,
            proximity_session_id="stale-session",
            observation=_observation(
                visit_id=4,
                track_id=7,
                marker_id=13,
                distance_mm=800,
                now_ms=1000,
            ),
        )

        status = coordinator.statuses(now_unix_milliseconds=1000)[0]
        self.assertEqual(status.state, "far")
        self.assertIsNone(status.proximity_session_id)

    def test_approach_dwell_no_duplicates_and_departure_rearm(self) -> None:
        coordinator = ShelfProximityCoordinator(_config())
        events = coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=7, distance_mm=800, now_ms=1000),),
            host_synced_seconds=1,
            now_unix_milliseconds=1000,
        )
        self.assertEqual(events, ())
        events = coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=7, distance_mm=790, now_ms=1500),),
            host_synced_seconds=1.5,
            now_unix_milliseconds=1500,
        )
        self.assertEqual([event.event_type for event in events], ["shelf_approach"])

        events = coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=11, distance_mm=780, now_ms=2200),),
            host_synced_seconds=2.2,
            now_unix_milliseconds=2200,
        )
        self.assertEqual(events, ())
        self.assertEqual(coordinator.statuses(now_unix_milliseconds=2200)[0].state, "near")

        coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=11, distance_mm=1200, now_ms=2500),),
            host_synced_seconds=2.5,
            now_unix_milliseconds=2500,
        )
        events = coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=11, distance_mm=1200, now_ms=3000),),
            host_synced_seconds=3,
            now_unix_milliseconds=3000,
        )
        self.assertEqual([event.event_type for event in events], ["shelf_departure"])

    def test_two_cameras_for_same_visit_do_not_duplicate(self) -> None:
        coordinator = ShelfProximityCoordinator(_config())
        coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=7, distance_mm=850, now_ms=1000),),
            host_synced_seconds=1,
            now_unix_milliseconds=1000,
        )
        coordinator.update_camera(
            camera_index=1,
            observations=(
                _observation(
                    visit_id=4,
                    track_id=2,
                    distance_mm=800,
                    now_ms=1200,
                    camera_index=1,
                ),
            ),
            host_synced_seconds=1.2,
            now_unix_milliseconds=1200,
        )
        events = coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=8, distance_mm=820, now_ms=1500),),
            host_synced_seconds=1.5,
            now_unix_milliseconds=1500,
        )
        self.assertEqual([event.event_type for event in events], ["shelf_approach"])
        events = coordinator.update_camera(
            camera_index=1,
            observations=(
                _observation(
                    visit_id=4,
                    track_id=3,
                    distance_mm=790,
                    now_ms=1800,
                    camera_index=1,
                ),
            ),
            host_synced_seconds=1.8,
            now_unix_milliseconds=1800,
        )
        self.assertEqual(events, ())

    def test_provisional_visit_does_not_emit(self) -> None:
        coordinator = ShelfProximityCoordinator(_config())
        observation = _observation(
            visit_id=4,
            track_id=7,
            distance_mm=700,
            now_ms=1000,
        )
        provisional = ShelfCameraObservation(
            **{**observation.__dict__, "visit_id": None}
        )
        coordinator.update_camera(
            camera_index=0,
            observations=(provisional,),
            host_synced_seconds=1,
            now_unix_milliseconds=1000,
        )
        events = coordinator.update_camera(
            camera_index=0,
            observations=(ShelfCameraObservation(**{**provisional.__dict__, "observed_at_unix_milliseconds": 2000}),),
            host_synced_seconds=2,
            now_unix_milliseconds=2000,
        )
        self.assertEqual(events, ())

    def test_close_visit_emits_departure_for_near_session(self) -> None:
        coordinator = ShelfProximityCoordinator(_config())
        coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=7, distance_mm=800, now_ms=1000),),
            host_synced_seconds=1,
            now_unix_milliseconds=1000,
        )
        coordinator.update_camera(
            camera_index=0,
            observations=(_observation(visit_id=4, track_id=7, distance_mm=800, now_ms=1500),),
            host_synced_seconds=1.5,
            now_unix_milliseconds=1500,
        )
        events = coordinator.close_visit(
            4,
            host_synced_seconds=2,
            now_unix_milliseconds=2000,
        )
        self.assertEqual([event.event_type for event in events], ["shelf_departure"])
        self.assertEqual(events[0].reason, "visit_closed")

    def test_one_visit_cannot_approach_two_ambiguous_shelves(self) -> None:
        shelf_one = _shelf(1, 10, "Shelf 1")
        shelf_two = _shelf(2, 11, "Shelf 2")
        coordinator = ShelfProximityCoordinator(
            replace(_config(), shelves=(shelf_one, shelf_two))
        )

        def observations(now_ms: int):
            return (
                _observation(
                    visit_id=4,
                    track_id=7,
                    shelf_id=1,
                    shelf_label="Shelf 1",
                    marker_id=10,
                    distance_mm=700,
                    now_ms=now_ms,
                ),
                _observation(
                    visit_id=4,
                    track_id=7,
                    shelf_id=2,
                    shelf_label="Shelf 2",
                    marker_id=11,
                    distance_mm=750,
                    now_ms=now_ms,
                ),
            )

        coordinator.update_camera(
            camera_index=0,
            observations=observations(1000),
            host_synced_seconds=1.0,
            now_unix_milliseconds=1000,
        )
        events = coordinator.update_camera(
            camera_index=0,
            observations=observations(1600),
            host_synced_seconds=1.6,
            now_unix_milliseconds=1600,
        )

        self.assertEqual(events, ())
        self.assertEqual(
            [status.state for status in coordinator.statuses(now_unix_milliseconds=1600)],
            ["far", "far"],
        )

    def test_active_shelf_blocks_second_shelf_for_same_visit(self) -> None:
        shelf_one = _shelf(1, 10, "Shelf 1")
        shelf_two = _shelf(2, 11, "Shelf 2")
        coordinator = ShelfProximityCoordinator(
            replace(_config(), shelves=(shelf_one, shelf_two))
        )
        shelf_one_observation = lambda now_ms, distance=700: _observation(
            visit_id=4,
            track_id=7,
            shelf_id=1,
            shelf_label="Shelf 1",
            marker_id=10,
            distance_mm=distance,
            now_ms=now_ms,
        )
        shelf_two_observation = lambda now_ms: _observation(
            visit_id=4,
            track_id=7,
            shelf_id=2,
            shelf_label="Shelf 2",
            marker_id=11,
            distance_mm=400,
            now_ms=now_ms,
        )
        coordinator.update_camera(
            camera_index=0,
            observations=(shelf_one_observation(1000),),
            host_synced_seconds=1.0,
            now_unix_milliseconds=1000,
        )
        events = coordinator.update_camera(
            camera_index=0,
            observations=(shelf_one_observation(1600),),
            host_synced_seconds=1.6,
            now_unix_milliseconds=1600,
        )
        self.assertEqual([event.shelf_id for event in events], [1])

        for now_ms in (1800, 2500):
            events = coordinator.update_camera(
                camera_index=0,
                observations=(
                    shelf_one_observation(now_ms, 800),
                    shelf_two_observation(now_ms),
                ),
                host_synced_seconds=now_ms / 1000,
                now_unix_milliseconds=now_ms,
            )
            self.assertEqual(events, ())

        statuses = coordinator.statuses(now_unix_milliseconds=2500)
        self.assertEqual(statuses[0].state, "near")
        self.assertEqual(statuses[1].state, "far")


if __name__ == "__main__":
    unittest.main()
