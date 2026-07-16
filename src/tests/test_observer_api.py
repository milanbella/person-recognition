import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.depth import DepthSample
from pipeline.observer_api import build_observer_camera_snapshot, observer_snapshot_payload
from pipeline.shop_state_store import ShopStateStore
from pipeline.tracking import Track
from pipeline.visit_identity import BodyAppearance, VisitAssignment
from pipeline.visit_registry import TrackVisitEvidence


class ObserverApiTests(unittest.TestCase):
    def test_builder_exposes_visible_current_frame_evidence(self) -> None:
        visible = Track(7, 10, 20, 110, 220, 0.84, status="TRACKED")
        lost = Track(8, 30, 40, 130, 240, 0.75, status="LOST")
        appearance = BodyAppearance(
            histogram=np.zeros((4,), dtype=np.float32),
            aspect_ratio=0.5,
            height_px=200,
        )
        evidence = TrackVisitEvidence(
            camera_role="observer",
            device_id="camera-a",
            track_id=7,
            host_seconds=12.5,
            track_bbox=(10, 20, 110, 220),
            face_identity_ids=("face_person_012",),
            body_appearance=appearance,
            depth_mm=3000.0,
        )
        assignment = VisitAssignment(
            visit_id=12,
            track_id=7,
            device_id="camera-a",
            face_identity_ids=("face_person_012",),
            matched_score=0.67,
            origin="entrance_confirmed",
        )
        depth = DepthSample(
            depth_mm=3000.0,
            valid_pixel_count=50,
            roi=(40, 100, 80, 180),
            anchor_px=(60, 180),
            point_3d_mm=(-100.0, 200.0, 3000.0),
        )

        snapshot = build_observer_camera_snapshot(
            camera_index=0,
            device_id="camera-a",
            camera_role="observer",
            rgb_frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            rgb_sequence_number=42,
            host_synced_seconds=12.5,
            tracks=[visible, lost],
            track_visit_evidence_by_id={7: evidence},
            visit_assignments={7: assignment},
            depth_samples={7: depth},
            customer_ids_by_visit={12: "customer-123"},
        )
        payload = observer_snapshot_payload(
            snapshot,
            age_milliseconds=10,
            status="active",
            include_observations=True,
        )

        self.assertEqual(len(payload["observations"]), 1)
        person = payload["observations"][0]
        self.assertEqual(person["trackId"], 7)
        self.assertEqual(person["visitId"], 12)
        self.assertEqual(person["customerId"], "customer-123")
        self.assertEqual(person["customerBindingStatus"], "bound")
        self.assertEqual(person["depth"]["validPixelCount"], 50)
        self.assertEqual(person["faceIdentityIds"], ["face_person_012"])
        self.assertEqual(person["visitMatch"]["state"], "matched")
        self.assertNotIn("histogram", person["body"])

    def test_pending_customer_and_missing_depth_are_explicit(self) -> None:
        track = Track(1, 1, 2, 11, 22, 0.9, status="NEW")
        assignment = VisitAssignment(
            visit_id=3,
            track_id=1,
            device_id="camera-a",
            face_identity_ids=(),
            matched_score=None,
            origin="entrance_confirmed",
        )
        snapshot = build_observer_camera_snapshot(
            camera_index=0,
            device_id="camera-a",
            camera_role="entrance_observer",
            rgb_frame=np.zeros((100, 200, 3), dtype=np.uint8),
            rgb_sequence_number=1,
            host_synced_seconds=1.0,
            tracks=[track],
            track_visit_evidence_by_id={},
            visit_assignments={1: assignment},
            depth_samples={},
            customer_ids_by_visit={},
        )
        person = observer_snapshot_payload(
            snapshot,
            age_milliseconds=0,
            status="active",
            include_observations=True,
        )["observations"][0]

        self.assertIsNone(person["customerId"])
        self.assertEqual(person["customerBindingStatus"], "pending")
        self.assertIsNone(person["depth"])
        self.assertEqual(person["visitMatch"]["state"], "matched")

    def test_provisional_match_state_is_additive_and_visit_remains_null(self) -> None:
        track = Track(9, 1, 2, 11, 22, 0.9, status="NEW")
        snapshot = build_observer_camera_snapshot(
            camera_index=4,
            device_id="remote-room",
            camera_role="observer",
            rgb_frame=np.zeros((100, 200, 3), dtype=np.uint8),
            rgb_sequence_number=2,
            host_synced_seconds=2.0,
            tracks=[track],
            track_visit_evidence_by_id={},
            visit_assignments={},
            depth_samples={},
            customer_ids_by_visit={},
            provisional_track_ids={9},
        )

        person = observer_snapshot_payload(
            snapshot,
            age_milliseconds=0,
            status="active",
            include_observations=True,
        )["observations"][0]

        self.assertIsNone(person["visitId"])
        self.assertEqual(person["visitMatch"]["state"], "provisional")

    def test_shop_customer_bindings_reload_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ShopStateStore(Path(temp_dir) / "shop.sqlite")
            store.record_entry(
                visit_id=4,
                host_seconds=1.0,
                device_id="camera-a",
                camera_role="entrance_observer",
                track_id=2,
                depth_mm=2000.0,
                plane_signed_distance_mm=-10.0,
                reason="test",
                event_payload={"type": "test"},
            )
            store.record_shop_customer_binding(visit_id=4, shopping_customer_id="customer-4")
            store.close()

            reopened = ShopStateStore(Path(temp_dir) / "shop.sqlite")
            self.assertEqual(reopened.load_shop_customer_bindings(), {4: "customer-4"})
            reopened.close()


if __name__ == "__main__":
    unittest.main()
