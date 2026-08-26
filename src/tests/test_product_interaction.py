import unittest
from dataclasses import replace

import numpy as np

from pipeline.depth import CameraIntrinsics
from pipeline.pose import PoseLandmark, PoseObservation
from pipeline.product_detection import ProductDetection
from pipeline.product_interaction import (
    EVENT_PRODUCT_PICKED,
    EVENT_PRODUCT_RETURNED,
    ClassAwareProductTracker,
    ProductDepthObservation,
    ProductInteractionSample,
    ProductInteractionCoordinator,
    ProductInteractionStateMachine,
    SHELF_OCCUPANCY_ON,
    SHELF_OCCUPANCY_OUTSIDE,
    SHELF_OCCUPANCY_UNKNOWN,
    ShelfDepthReference,
    hand_product_association,
    product_shelf_depth_reference,
    sample_product_depth,
)
from pipeline.shelf_regions import NormalizedPoint, ShelfDepthCell, ShelfDepthModel, ShelfRegion


def _detection(x1: int, *, class_id: int = 1) -> ProductDetection:
    return ProductDetection(x1, 100, x1 + 100, 300, 0.9, class_id, "oil")


def _sample(
    time_seconds: float,
    *,
    shelf_id: int | None,
    hand_score: float,
    product_track_id: int = 1,
    visit_id: int = 4,
    person_track_id: int = 7,
    occupancy: str | None = None,
) -> ProductInteractionSample:
    resolved_occupancy = occupancy or (
        SHELF_OCCUPANCY_ON if shelf_id is not None else SHELF_OCCUPANCY_OUTSIDE
    )
    on_shelf = resolved_occupancy == SHELF_OCCUPANCY_ON
    point = (100.0, 100.0, 4000.0) if on_shelf else (400.0, 100.0, 3400.0)
    depth = ProductDepthObservation(
        rgb_sequence_number=int(time_seconds * 10),
        depth_sequence_number=int(time_seconds * 10),
        timestamp_delta_milliseconds=0.0,
        roi=(10, 10, 20, 20),
        valid_pixel_count=100,
        valid_fraction=1.0,
        median_depth_mm=point[2],
        mad_depth_mm=10.0,
        point_3d_mm=point,
    )
    return ProductInteractionSample(
        camera_index=0,
        device_id="camera-a",
        person_track_id=person_track_id,
        product_track_id=product_track_id,
        visit_id=visit_id,
        customer_id="customer-4",
        product_class_id=1,
        product_label="oil",
        product_score=0.9,
        product_box=(100, 100, 200, 300),
        shelf_id=shelf_id,
        hand="left" if hand_score else None,
        hand_score=hand_score,
        host_synced_seconds=time_seconds,
        rgb_sequence_number=int(time_seconds * 10),
        product_depth=depth,
        shelf_depth_reference=ShelfDepthReference(
            shelf_id=shelf_id,
            occupancy=resolved_occupancy,
            expected_depth_mm=4000.0 if shelf_id is not None else None,
            depth_residual_mm=(point[2] - 4000.0) if shelf_id is not None else None,
        ),
    )


class ProductInteractionTests(unittest.TestCase):
    @staticmethod
    def _pose(time_seconds: float) -> PoseObservation:
        return PoseObservation(
            camera_index=0,
            device_id="camera-a",
            track_id=7,
            rgb_sequence_number=int(time_seconds * 10),
            host_synced_seconds=time_seconds,
            observed_at_unix_milliseconds=1,
            inference_milliseconds=1,
            person_box=(0, 0, 100, 200),
            pose_box=(0, 0, 100, 200),
            landmarks=(),
        )

    def test_coordinator_retains_pose_history_for_delayed_depth_join(self) -> None:
        coordinator = ProductInteractionCoordinator()
        expected = self._pose(10.0)
        coordinator.publish_pose(expected)
        coordinator.publish_pose(self._pose(13.0))

        self.assertIs(coordinator.nearest_pose(0, 7, 10.1), expected)
        self.assertIsNone(coordinator.nearest_pose(0, 7, 11.0))

    @staticmethod
    def _shelf_region() -> ShelfRegion:
        model = ShelfDepthModel(
            1, 1, 8, 8, 0.35,
            (ShelfDepthCell(0, 0, 4000.0, 10.0, 64, 1.0),),
            10, 1, 8.0, 8.0, 4.0, 4.0, "calibration-a",
        )
        return ShelfRegion(
            3,
            (
                NormalizedPoint(0.0, 0.0),
                NormalizedPoint(1.0, 0.0),
                NormalizedPoint(1.0, 1.0),
                NormalizedPoint(0.0, 1.0),
            ),
            250.0,
            model,
            150.0,
        )

    def test_product_tracker_preserves_class_aware_id(self) -> None:
        tracker = ClassAwareProductTracker()
        first = tracker.update([_detection(100)])[0]
        second = tracker.update([_detection(115)])[0]
        other_class = tracker.update([_detection(120, class_id=2)])

        self.assertEqual(first.product_track_id, second.product_track_id)
        self.assertNotEqual(second.product_track_id, other_class[0].product_track_id)

    def test_hand_association_uses_visible_wrist(self) -> None:
        pose = PoseObservation(
            camera_index=0,
            device_id="camera-a",
            track_id=7,
            rgb_sequence_number=1,
            host_synced_seconds=1.0,
            observed_at_unix_milliseconds=1,
            inference_milliseconds=5,
            person_box=(0, 0, 500, 700),
            pose_box=(0, 0, 500, 700),
            landmarks=(
                PoseLandmark("left_wrist", 0.15, 0.25, 0.9),
                PoseLandmark("right_wrist", 0.9, 0.9, 0.9),
            ),
            source_frame_width=1000,
            source_frame_height=800,
        )
        hand, score = hand_product_association(
            _detection(100),
            pose,
            frame_width=1000,
            frame_height=800,
            keypoint_threshold=0.35,
        )

        self.assertEqual(hand, "left")
        self.assertGreater(score, 0.5)

    def test_hand_association_scales_processing_person_box_to_raw_frame(self) -> None:
        pose = PoseObservation(
            camera_index=0,
            device_id="camera-a",
            track_id=7,
            rgb_sequence_number=1,
            host_synced_seconds=1.0,
            observed_at_unix_milliseconds=1,
            inference_milliseconds=5,
            person_box=(0, 0, 500, 700),
            pose_box=(0, 0, 500, 700),
            landmarks=(PoseLandmark("left_wrist", 0.15, 0.25, 0.9),),
            source_frame_width=1000,
            source_frame_height=800,
        )
        detection = ProductDetection(200, 200, 400, 600, 0.9, 1, "oil")

        hand, score = hand_product_association(
            detection,
            pose,
            frame_width=2000,
            frame_height=1600,
            keypoint_threshold=0.35,
        )

        self.assertEqual(hand, "left")
        self.assertAlmostEqual(score, 0.9)

    def test_pick_and_return_require_temporal_transitions(self) -> None:
        machine = ProductInteractionStateMachine(
            shelf_stable_seconds=0.4,
            outside_confirm_seconds=0.2,
            return_confirm_seconds=0.3,
        )
        self.assertIsNone(machine.update(_sample(0.0, shelf_id=3, hand_score=0.0)))
        self.assertIsNone(machine.update(_sample(0.5, shelf_id=3, hand_score=0.8)))
        self.assertIsNone(machine.update(_sample(0.6, shelf_id=None, hand_score=0.8)))
        picked = machine.update(_sample(0.8, shelf_id=None, hand_score=0.8))

        assert picked is not None
        self.assertEqual(picked.event_type, EVENT_PRODUCT_PICKED)
        self.assertEqual(picked.shelf_id, 3)

        self.assertIsNone(machine.update(_sample(1.0, shelf_id=3, hand_score=0.8)))
        self.assertIsNone(machine.update(_sample(1.1, shelf_id=3, hand_score=0.0)))
        returned = machine.update(_sample(1.5, shelf_id=3, hand_score=0.0))

        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)
        self.assertEqual(returned.shelf_id, 3)

    def test_reach_without_removing_product_emits_no_event(self) -> None:
        machine = ProductInteractionStateMachine(shelf_stable_seconds=0.2)
        for sample in (
            _sample(0.0, shelf_id=2, hand_score=0.0),
            _sample(0.3, shelf_id=2, hand_score=0.8),
            _sample(0.6, shelf_id=2, hand_score=0.0),
        ):
            self.assertIsNone(machine.update(sample))

    def test_return_does_not_require_an_earlier_pick(self) -> None:
        machine = ProductInteractionStateMachine(return_confirm_seconds=0.3)
        self.assertIsNone(machine.update(_sample(0.0, shelf_id=None, hand_score=0.8)))
        self.assertIsNone(machine.update(_sample(0.2, shelf_id=3, hand_score=0.8)))
        self.assertIsNone(machine.update(_sample(0.3, shelf_id=3, hand_score=0.0)))
        returned = machine.update(_sample(0.7, shelf_id=3, hand_score=0.0))

        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)
        self.assertEqual(returned.visit_id, 4)
        self.assertGreater(returned.confidence, 0.0)

    def test_return_carry_context_transfers_to_another_visit(self) -> None:
        coordinator = ProductInteractionCoordinator()
        carried = _sample(
            0.0,
            shelf_id=3,
            hand_score=0.8,
            visit_id=4,
            person_track_id=7,
            occupancy="OFF_SHELF_3D",
        )
        coordinator._apply_carry_context(carried)
        self.assertIsNone(coordinator.state_machine.update(carried))

        placing = _sample(
            0.6,
            shelf_id=3,
            hand_score=0.8,
            product_track_id=2,
            visit_id=5,
            person_track_id=8,
        )
        coordinator._apply_carry_context(placing)
        self.assertIsNone(coordinator.state_machine.update(placing))
        self.assertIsNone(
            coordinator.state_machine.update(
                _sample(
                    0.7,
                    shelf_id=3,
                    hand_score=0.0,
                    product_track_id=2,
                    visit_id=5,
                    person_track_id=8,
                )
            )
        )
        returned = coordinator.state_machine.update(
            _sample(
                1.2,
                shelf_id=3,
                hand_score=0.0,
                product_track_id=2,
                visit_id=5,
                person_track_id=8,
            )
        )

        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)
        self.assertEqual(returned.visit_id, 5)

    def test_return_transfers_across_delayed_visit_and_product_track_split(self) -> None:
        coordinator = ProductInteractionCoordinator()
        carried = _sample(
            0.0,
            shelf_id=4,
            hand_score=0.8,
            visit_id=7,
            person_track_id=1,
            occupancy="OFF_SHELF_3D",
        )
        coordinator._apply_carry_context(carried)
        self.assertIsNone(coordinator.state_machine.update(carried))

        placing = _sample(
            4.0,
            shelf_id=4,
            hand_score=0.8,
            product_track_id=2,
            visit_id=8,
            person_track_id=3,
        )
        coordinator._apply_carry_context(placing)
        self.assertIsNone(coordinator.state_machine.update(placing))

        released = _sample(
            4.6,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=3,
            visit_id=8,
            person_track_id=3,
        )
        coordinator._apply_carry_context(released)
        self.assertIsNone(coordinator.state_machine.update(released))

        stable_release = _sample(
            5.1,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=4,
            visit_id=8,
            person_track_id=3,
        )
        returned = coordinator._apply_carry_context(stable_release)

        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)
        self.assertEqual(returned.visit_id, 8)
        self.assertEqual(returned.product_track_id, 4)

    def test_return_survives_interleaved_carry_and_repeated_placement_track_churn(
        self,
    ) -> None:
        coordinator = ProductInteractionCoordinator()
        for time_seconds in (0.0, 0.3):
            carried = _sample(
                time_seconds,
                shelf_id=3,
                hand_score=0.81,
                product_track_id=2,
                occupancy="OFF_SHELF_3D",
            )
            coordinator._apply_carry_context(carried)
            self.assertIsNone(coordinator.state_machine.update(carried))

            shelf_product = _sample(
                time_seconds,
                shelf_id=3,
                hand_score=0.79,
                product_track_id=3,
            )
            coordinator._apply_carry_context(shelf_product)
            self.assertIsNone(coordinator.state_machine.update(shelf_product))

        for time_seconds, product_track_id, hand_score in (
            (0.9, 4, 0.82),
            (1.2, 4, 0.82),
            (1.5, 5, 0.0),
        ):
            placement = _sample(
                time_seconds,
                shelf_id=3,
                hand_score=hand_score,
                product_track_id=product_track_id,
            )
            self.assertIsNone(coordinator._apply_carry_context(placement))
            self.assertIsNone(coordinator.state_machine.update(placement))

        stable_placement = _sample(
            2.0,
            shelf_id=3,
            hand_score=0.0,
            product_track_id=6,
        )
        returned = coordinator._apply_carry_context(stable_placement)
        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)
        self.assertEqual(returned.shelf_id, 3)
        self.assertEqual(returned.product_track_id, 6)

    def test_active_pick_suppresses_cross_camera_duplicates_until_return(self) -> None:
        coordinator = ProductInteractionCoordinator()
        emitted = []
        first_pick = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            _sample(0.0, shelf_id=3, hand_score=0.8),
            3,
        )
        coordinator._accept_event(first_pick, emitted)

        duplicate_pick = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            replace(
                _sample(4.0, shelf_id=3, hand_score=0.7),
                camera_index=1,
                device_id="camera-b",
            ),
            3,
        )
        coordinator._accept_event(duplicate_pick, emitted)
        self.assertEqual([event.event_type for event in emitted], [EVENT_PRODUCT_PICKED])

        returned = coordinator.state_machine.confirm_return(
            replace(
                _sample(10.0, shelf_id=3, hand_score=0.0),
                camera_index=1,
                device_id="camera-b",
            ),
            carry_contact_score=0.8,
        )
        coordinator._accept_event(returned, emitted)
        self.assertEqual(
            [event.event_type for event in emitted],
            [EVENT_PRODUCT_PICKED, EVENT_PRODUCT_RETURNED],
        )

        cooldown_pick = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            _sample(12.0, shelf_id=3, hand_score=0.8),
            3,
        )
        coordinator._accept_event(cooldown_pick, emitted)
        self.assertEqual(len(emitted), 2)

    def test_active_pick_can_return_on_another_camera_after_track_churn(self) -> None:
        coordinator = ProductInteractionCoordinator()
        emitted = []
        picked = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            _sample(0.0, shelf_id=3, hand_score=0.8),
            3,
        )
        coordinator._accept_event(picked, emitted)

        for time_seconds, product_track_id, hand_score in (
            (2.0, 3, 0.75),
            (2.2, 4, 0.30),
        ):
            placement = replace(
                _sample(
                    time_seconds,
                    shelf_id=3,
                    hand_score=hand_score,
                    product_track_id=product_track_id,
                ),
                camera_index=1,
                device_id="camera-b",
            )
            self.assertIsNone(coordinator._apply_carry_context(placement))
            self.assertIsNone(coordinator.state_machine.update(placement))

        stable_placement = replace(
            _sample(2.7, shelf_id=3, hand_score=0.0, product_track_id=5),
            camera_index=1,
            device_id="camera-b",
        )
        returned = coordinator._apply_carry_context(stable_placement)
        assert returned is not None
        coordinator._accept_event(returned, emitted)

        self.assertEqual(
            [event.event_type for event in emitted],
            [EVENT_PRODUCT_PICKED, EVENT_PRODUCT_RETURNED],
        )
        self.assertEqual(emitted[-1].camera_index, 1)

    def test_pick_transfers_shelf_contact_across_product_track_split(self) -> None:
        coordinator = ProductInteractionCoordinator()
        first_shelf_contact = _sample(
            0.0,
            shelf_id=4,
            hand_score=0.8,
            product_track_id=1,
        )
        coordinator._apply_carry_context(first_shelf_contact)
        self.assertIsNone(coordinator.state_machine.update(first_shelf_contact))

        stable_shelf_contact = _sample(
            0.5,
            shelf_id=4,
            hand_score=0.8,
            product_track_id=1,
        )
        coordinator._apply_carry_context(stable_shelf_contact)
        self.assertIsNone(coordinator.state_machine.update(stable_shelf_contact))

        first_departure = _sample(
            0.6,
            shelf_id=None,
            hand_score=0.8,
            product_track_id=2,
        )
        coordinator._apply_carry_context(first_departure)
        self.assertIsNone(coordinator.state_machine.update(first_departure))

        confirmed_departure = _sample(
            0.9,
            shelf_id=None,
            hand_score=0.8,
            product_track_id=2,
        )
        coordinator._apply_carry_context(confirmed_departure)
        picked = coordinator.state_machine.update(confirmed_departure)

        assert picked is not None
        self.assertEqual(picked.event_type, EVENT_PRODUCT_PICKED)
        self.assertEqual(picked.shelf_id, 4)

    def test_pick_transfers_stable_shelf_presence_when_contact_starts_after_split(self) -> None:
        coordinator = ProductInteractionCoordinator()
        for time_seconds in (0.0, 0.5):
            shelf_product = _sample(
                time_seconds,
                shelf_id=3,
                hand_score=0.0,
                product_track_id=1,
            )
            coordinator._apply_carry_context(shelf_product)
            self.assertIsNone(coordinator.state_machine.update(shelf_product))

        first_departure = _sample(
            1.0,
            shelf_id=None,
            hand_score=0.8,
            product_track_id=2,
        )
        coordinator._apply_carry_context(first_departure)
        self.assertIsNone(coordinator.state_machine.update(first_departure))

        confirmed_departure = _sample(
            1.3,
            shelf_id=None,
            hand_score=0.8,
            product_track_id=2,
        )
        coordinator._apply_carry_context(confirmed_departure)
        picked = coordinator.state_machine.update(confirmed_departure)

        assert picked is not None
        self.assertEqual(picked.event_type, EVENT_PRODUCT_PICKED)
        self.assertEqual(picked.shelf_id, 3)

    def test_pick_handoff_overrides_brief_on_shelf_state_on_new_track(self) -> None:
        coordinator = ProductInteractionCoordinator()
        for time_seconds in (0.0, 0.5):
            stable_shelf_product = _sample(
                time_seconds,
                shelf_id=4,
                hand_score=0.0,
                product_track_id=1,
            )
            coordinator._apply_carry_context(stable_shelf_product)
            self.assertIsNone(
                coordinator.state_machine.update(stable_shelf_product)
            )

        new_track_on_shelf = _sample(
            0.6,
            shelf_id=4,
            hand_score=0.6,
            product_track_id=2,
        )
        coordinator._apply_carry_context(new_track_on_shelf)
        self.assertIsNone(coordinator.state_machine.update(new_track_on_shelf))

        first_departure = _sample(
            0.7,
            shelf_id=4,
            hand_score=0.8,
            product_track_id=2,
            occupancy="OFF_SHELF_3D",
        )
        coordinator._apply_carry_context(first_departure)
        self.assertIsNone(coordinator.state_machine.update(first_departure))

        confirmed_departure = _sample(
            1.0,
            shelf_id=4,
            hand_score=0.8,
            product_track_id=2,
            occupancy="OFF_SHELF_3D",
        )
        coordinator._apply_carry_context(confirmed_departure)
        picked = coordinator.state_machine.update(confirmed_departure)

        assert picked is not None
        self.assertEqual(picked.event_type, EVENT_PRODUCT_PICKED)
        self.assertEqual(picked.shelf_id, 4)

    def test_pick_from_one_shelf_can_return_to_another_shelf_and_camera(self) -> None:
        coordinator = ProductInteractionCoordinator()
        emitted = []
        picked = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            _sample(0.0, shelf_id=3, hand_score=0.8),
            3,
        )
        coordinator._accept_event(picked, emitted)

        placing = replace(
            _sample(2.0, shelf_id=4, hand_score=0.75, product_track_id=3),
            camera_index=1,
            device_id="camera-b",
            visit_id=9,
        )
        self.assertIsNone(coordinator._apply_carry_context(placing))

        first_release = replace(
            _sample(2.2, shelf_id=4, hand_score=0.0, product_track_id=4),
            camera_index=2,
            device_id="camera-c",
            visit_id=10,
        )
        self.assertIsNone(coordinator._apply_carry_context(first_release))

        stable_release = replace(
            _sample(2.7, shelf_id=4, hand_score=0.0, product_track_id=4),
            camera_index=2,
            device_id="camera-c",
            visit_id=10,
        )
        returned = coordinator._apply_carry_context(stable_release)
        assert returned is not None
        coordinator._accept_event(returned, emitted)

        self.assertEqual(
            [(event.event_type, event.shelf_id) for event in emitted],
            [(EVENT_PRODUCT_PICKED, 3), (EVENT_PRODUCT_RETURNED, 4)],
        )
        self.assertEqual(emitted[-1].visit_id, 10)

    def test_release_view_is_not_reset_by_other_camera_hand_overlap(self) -> None:
        coordinator = ProductInteractionCoordinator()
        emitted = []
        picked = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            _sample(0.0, shelf_id=3, hand_score=0.8),
            3,
        )
        coordinator._accept_event(picked, emitted)

        placing = replace(
            _sample(2.0, shelf_id=4, hand_score=0.75, product_track_id=2),
            camera_index=0,
            device_id="camera-a",
        )
        self.assertIsNone(coordinator._apply_carry_context(placing))

        release_view = replace(
            _sample(2.1, shelf_id=4, hand_score=0.1, product_track_id=3),
            camera_index=2,
            device_id="camera-c",
        )
        self.assertIsNone(coordinator._apply_carry_context(release_view))

        obscured_view = replace(
            _sample(2.4, shelf_id=4, hand_score=0.7, product_track_id=4),
            camera_index=1,
            device_id="camera-b",
        )
        self.assertIsNone(coordinator._apply_carry_context(obscured_view))

        stable_release = replace(release_view, host_synced_seconds=2.6)
        returned = coordinator._apply_carry_context(stable_release)
        assert returned is not None
        coordinator._accept_event(returned, emitted)

        self.assertEqual(
            [(event.event_type, event.shelf_id) for event in emitted],
            [(EVENT_PRODUCT_PICKED, 3), (EVENT_PRODUCT_RETURNED, 4)],
        )

    def test_release_track_is_not_reset_by_same_camera_duplicate_track(self) -> None:
        coordinator = ProductInteractionCoordinator()
        emitted = []
        picked = coordinator.state_machine._event(
            EVENT_PRODUCT_PICKED,
            _sample(0.0, shelf_id=4, hand_score=0.8),
            4,
        )
        coordinator._accept_event(picked, emitted)

        placing = _sample(
            2.0,
            shelf_id=3,
            hand_score=0.7,
            product_track_id=5,
        )
        self.assertIsNone(coordinator._apply_carry_context(placing))

        release_track = _sample(
            2.1,
            shelf_id=3,
            hand_score=0.05,
            product_track_id=4,
        )
        self.assertIsNone(coordinator._apply_carry_context(release_track))

        overlapping_duplicate = replace(placing, host_synced_seconds=2.4)
        self.assertIsNone(
            coordinator._apply_carry_context(overlapping_duplicate)
        )

        stable_release = replace(release_track, host_synced_seconds=2.6)
        returned = coordinator._apply_carry_context(stable_release)
        assert returned is not None
        coordinator._accept_event(returned, emitted)

        self.assertEqual(
            [(event.event_type, event.shelf_id) for event in emitted],
            [(EVENT_PRODUCT_PICKED, 4), (EVENT_PRODUCT_RETURNED, 3)],
        )

    def test_one_frame_shelf_contact_does_not_transfer_to_pick(self) -> None:
        coordinator = ProductInteractionCoordinator()
        shelf_contact = _sample(
            0.0,
            shelf_id=1,
            hand_score=0.8,
            product_track_id=1,
        )
        coordinator._apply_carry_context(shelf_contact)
        self.assertIsNone(coordinator.state_machine.update(shelf_contact))

        for time_seconds in (0.1, 0.5):
            departure = _sample(
                time_seconds,
                shelf_id=None,
                hand_score=0.8,
                product_track_id=2,
            )
            coordinator._apply_carry_context(departure)
            self.assertIsNone(coordinator.state_machine.update(departure))

    def test_contacted_shelf_product_disappearance_confirms_pick(self) -> None:
        machine = ProductInteractionStateMachine(
            shelf_stable_seconds=0.4,
            outside_confirm_seconds=0.25,
            missing_confirm_seconds=0.75,
        )
        shelf = _sample(0.0, shelf_id=3, hand_score=0.0)
        contact = _sample(0.5, shelf_id=3, hand_score=0.76)
        self.assertIsNone(machine.update(shelf))
        self.assertIsNone(machine.update(contact))
        self.assertIsNone(machine.update_missing(contact, 1.2))

        picked = machine.update_missing(contact, 1.3)
        assert picked is not None
        self.assertEqual(picked.event_type, EVENT_PRODUCT_PICKED)
        self.assertEqual(picked.shelf_id, 3)
        self.assertAlmostEqual(picked.occurred_host_synced_seconds, 1.3)

    def test_return_accepts_stable_shelf_placement_after_contact_weakens(self) -> None:
        machine = ProductInteractionStateMachine(return_confirm_seconds=0.4)
        carried = _sample(
            0.0,
            shelf_id=3,
            hand_score=0.8,
            occupancy="OFF_SHELF_3D",
        )
        placing = _sample(0.5, shelf_id=3, hand_score=0.7)
        weak_contact = _sample(0.8, shelf_id=3, hand_score=0.33)
        stable = _sample(1.3, shelf_id=3, hand_score=0.33)
        self.assertIsNone(machine.update(carried))
        self.assertIsNone(machine.update(placing))
        self.assertIsNone(machine.update(weak_contact))

        returned = machine.update(stable)
        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)
        self.assertEqual(returned.shelf_id, 3)

    def test_visible_carry_prevents_static_shelf_product_return(self) -> None:
        coordinator = ProductInteractionCoordinator()
        for time_seconds in (0.0, 0.55, 1.1):
            carried = _sample(
                time_seconds,
                shelf_id=4,
                hand_score=0.8,
                product_track_id=1,
                occupancy="OFF_SHELF_3D",
            )
            coordinator._apply_carry_context(carried)
            self.assertIsNone(coordinator.state_machine.update(carried))

            shelf_product = _sample(
                time_seconds + 0.1,
                shelf_id=4,
                hand_score=0.0,
                product_track_id=2,
            )
            coordinator._apply_carry_context(shelf_product)
            self.assertIsNone(coordinator.state_machine.update(shelf_product))

        first_return_frame = _sample(
            1.7,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=3,
        )
        coordinator._apply_carry_context(first_return_frame)
        self.assertIsNone(coordinator.state_machine.update(first_return_frame))

        stable_return_frame = _sample(
            2.2,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=3,
        )
        coordinator._apply_carry_context(stable_return_frame)
        returned = coordinator.state_machine.update(stable_return_frame)
        assert returned is not None
        self.assertEqual(returned.event_type, EVENT_PRODUCT_RETURNED)

    def test_completed_return_clears_competing_product_track_hypotheses(self) -> None:
        coordinator = ProductInteractionCoordinator()
        carried = _sample(
            0.0,
            shelf_id=4,
            hand_score=0.8,
            occupancy="OFF_SHELF_3D",
        )
        coordinator._apply_carry_context(carried)
        self.assertIsNone(coordinator.state_machine.update(carried))

        for product_track_id in (2, 3):
            placing = _sample(
                0.6,
                shelf_id=4,
                hand_score=0.8,
                product_track_id=product_track_id,
            )
            coordinator._apply_carry_context(placing)
            self.assertIsNone(coordinator.state_machine.update(placing))

        first_release = _sample(
            1.2,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=2,
        )
        coordinator._apply_carry_context(first_release)
        self.assertIsNone(coordinator.state_machine.update(first_release))

        stable_release = _sample(
            1.7,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=4,
        )
        returned = coordinator._apply_carry_context(stable_release)
        assert returned is not None
        coordinator.state_machine.reset_product_class(returned.product_class_id)
        coordinator._clear_interaction_context(returned.product_class_id)

        competing_release = _sample(
            1.3,
            shelf_id=4,
            hand_score=0.0,
            product_track_id=3,
        )
        coordinator._apply_carry_context(competing_release)
        self.assertIsNone(coordinator.state_machine.update(competing_release))

    def test_missing_depth_never_advances_pick_state(self) -> None:
        machine = ProductInteractionStateMachine(shelf_stable_seconds=0.2)
        self.assertIsNone(machine.update(_sample(0.0, shelf_id=2, hand_score=0.0)))
        self.assertIsNone(machine.update(_sample(0.3, shelf_id=2, hand_score=0.8)))
        missing = _sample(0.8, shelf_id=None, hand_score=0.8)
        missing = ProductInteractionSample(
            **{
                **missing.__dict__,
                "product_depth": None,
                "shelf_depth_reference": ShelfDepthReference(
                    shelf_id=None,
                    occupancy=SHELF_OCCUPANCY_UNKNOWN,
                    suppression_reason="invalid_product_depth",
                ),
            }
        )

        self.assertIsNone(machine.update(missing))
        self.assertIsNone(machine.update(missing))

    def test_product_depth_classifies_on_and_off_shelf_in_3d(self) -> None:
        detection = ProductDetection(2, 2, 6, 6, 0.9, 1, "oil")
        intrinsics = CameraIntrinsics(8.0, 8.0, 4.0, 4.0)
        on_depth = sample_product_depth(
            np.full((8, 8), 4000, dtype=np.uint16),
            detection,
            source_frame_width=8,
            source_frame_height=8,
            rgb_sequence_number=1,
            depth_sequence_number=1,
            timestamp_delta_milliseconds=0.0,
            intrinsics=intrinsics,
            minimum_valid_pixels=4,
        )
        off_depth = sample_product_depth(
            np.full((8, 8), 3400, dtype=np.uint16),
            detection,
            source_frame_width=8,
            source_frame_height=8,
            rgb_sequence_number=2,
            depth_sequence_number=2,
            timestamp_delta_milliseconds=0.0,
            intrinsics=intrinsics,
            minimum_valid_pixels=4,
        )

        assert on_depth is not None and off_depth is not None
        on_reference = product_shelf_depth_reference(
            detection,
            source_frame_width=8,
            source_frame_height=8,
            shelf_regions={3: self._shelf_region()},
            product_depth=on_depth,
        )
        off_reference = product_shelf_depth_reference(
            detection,
            source_frame_width=8,
            source_frame_height=8,
            shelf_regions={3: self._shelf_region()},
            product_depth=off_depth,
        )

        self.assertEqual(on_reference.occupancy, SHELF_OCCUPANCY_ON)
        self.assertEqual(off_reference.occupancy, "OFF_SHELF_3D")
        self.assertEqual(off_reference.depth_residual_mm, -600.0)


if __name__ == "__main__":
    unittest.main()
