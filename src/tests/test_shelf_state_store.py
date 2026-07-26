import tempfile
import unittest
from pathlib import Path

from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_proximity import ShelfProximityEvent
from pipeline.shop_state_store import ShopStateStore


def _event(event_type: str, *, session_id: str = "1:visit-4:1000") -> ShelfProximityEvent:
    return ShelfProximityEvent(
        event_type=event_type,
        proximity_session_id=session_id,
        shelf_id=1,
        shelf_label="Drinks",
        marker_id=10,
        visit_id=4,
        visit_origin="entrance_confirmed",
        customer_id="customer-4",
        camera_index=2,
        device_id="camera-a",
        track_id=7,
        distance_mm=800 if event_type == "shelf_approach" else 1200,
        threshold_mm=900 if event_type == "shelf_approach" else 1100,
        host_synced_seconds=1.0,
        occurred_at_unix_milliseconds=1000,
        rgb_sequence_number=12,
        depth_sequence_number=12,
        person_point_3d_mm=(0, 0, 2200),
        anchor=ShelfAnchor(
            shelf_id=1,
            marker_id=10,
            device_id="camera-a",
            point_3d_mm=(0, 0, 3000),
            sample_count=20,
            rms_spread_mm=4,
            updated_at_unix_milliseconds=900,
            source="operator_calibrated",
        ),
        reason="distance_dwell",
    )


class ShelfStateStoreTests(unittest.TestCase):
    def test_persists_idempotent_events_and_active_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ShopStateStore(Path(temporary) / "state.sqlite")
            try:
                approach = store.record_shelf_event(_event("shelf_approach"))
                duplicate = store.record_shelf_event(_event("shelf_approach"))
                self.assertEqual(approach.event_id, duplicate.event_id)
                self.assertEqual(len(store.load_recent_shelf_event_payloads()), 1)
                active = store.load_active_shelf_session_payloads()
                self.assertEqual(len(active), 1)
                self.assertEqual(active[0]["visitId"], 4)

                departure = store.record_shelf_event(_event("shelf_departure"))
                self.assertIsNotNone(departure.event_id)
                self.assertEqual(store.load_active_shelf_session_payloads(), [])
                self.assertEqual(len(store.load_recent_shelf_event_payloads()), 2)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
