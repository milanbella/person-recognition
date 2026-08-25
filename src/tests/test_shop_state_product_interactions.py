import tempfile
import unittest
from pathlib import Path

from pipeline.product_interaction import ProductInteractionEvent
from pipeline.shop_state_store import ShopStateStore


class ShopStateProductInteractionTests(unittest.TestCase):
    def test_event_is_persisted_idempotently_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ShopStateStore(Path(directory) / "state.sqlite")
            try:
                event = ProductInteractionEvent(
                    event_id="event-1",
                    event_type="PRODUCT_PICKED",
                    visit_id=4,
                    customer_id=None,
                    product_class_id=5,
                    product_label="water",
                    shelf_id=3,
                    occurred_host_synced_seconds=18.25,
                    confidence=0.61,
                    camera_index=0,
                    device_id="camera-a",
                    person_track_id=2,
                    product_track_id=8,
                    rgb_sequence_number=99,
                    hand="right",
                    hand_score=0.75,
                    pose_sequence_number=98,
                    pose_delta_milliseconds=33,
                )
                persisted_id = store.record_product_interaction_event(event)
                self.assertIsNotNone(persisted_id)
                self.assertIsNone(store.record_product_interaction_event(event))
                loaded = store.load_product_interaction_events()
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["id"], persisted_id)
                self.assertEqual(loaded[0]["evidence"]["hand"], "right")
                self.assertEqual(loaded[0]["status"], "candidate")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
