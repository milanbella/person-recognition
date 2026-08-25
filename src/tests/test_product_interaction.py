import unittest

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
) -> ProductInteractionSample:
    on_shelf = shelf_id is not None
    point = (100.0, 100.0, 4000.0) if on_shelf else (400.0, 100.0, 3600.0)
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
            occupancy=(SHELF_OCCUPANCY_ON if on_shelf else SHELF_OCCUPANCY_OUTSIDE),
            expected_depth_mm=4000.0 if on_shelf else None,
            depth_residual_mm=0.0 if on_shelf else None,
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
            shelf_id=None,
            hand_score=0.8,
            visit_id=4,
            person_track_id=7,
        )
        coordinator._apply_carry_context(carried)
        self.assertIsNone(coordinator.state_machine.update(carried))

        placing = _sample(
            0.2,
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
                    0.3,
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
                0.8,
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
