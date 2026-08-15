import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_position_push import ShelfPositionOutbox, ShelfPositionPushService
from pipeline.shelf_proximity import ShelfCameraObservation


def _observation(*, shelf_id: int, distance_mm: float) -> ShelfCameraObservation:
    anchor = ShelfAnchor(
        shelf_id=shelf_id,
        marker_id=10 + shelf_id,
        device_id="camera-a",
        point_3d_mm=(0.0, 0.0, 3000.0),
        sample_count=20,
        rms_spread_mm=4.0,
        updated_at_unix_milliseconds=900,
        source="operator_calibrated",
    )
    return ShelfCameraObservation(
        shelf_id=shelf_id,
        shelf_label=f"Shelf {shelf_id}",
        marker_id=anchor.marker_id,
        camera_index=0,
        device_id="camera-a",
        track_id=7,
        visit_id=4,
        visit_origin="entrance_confirmed",
        customer_id="customer-4",
        distance_mm=distance_mm,
        person_point_3d_mm=(0.0, 0.0, 2000.0),
        anchor=anchor,
        host_synced_seconds=1.0,
        observed_at_unix_milliseconds=1000,
        rgb_sequence_number=12,
        depth_sequence_number=12,
    )


class ShelfPositionPushTests(unittest.TestCase):
    def test_nearest_shelf_is_enqueued_then_cleared_after_stale_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
            service = ShelfPositionPushService(
                db_path=Path(temporary) / "state.sqlite",
                shop_id=1,
                sender=lambda _payload: None,
                stability_seconds=0.0,
                stale_seconds=2.0,
            )
            observations = {
                (0, 1, 4, 11): (_observation(shelf_id=1, distance_mm=800), 10.0),
                (0, 2, 4, 12): (_observation(shelf_id=2, distance_mm=500), 10.0),
            }
            candidates = {}
            last_enqueued = {}
            try:
                service._evaluate(
                    outbox=outbox,
                    observations=observations,
                    candidates=candidates,
                    last_enqueued=last_enqueued,
                    now=10.0,
                )
                service._evaluate(
                    outbox=outbox,
                    observations=observations,
                    candidates=candidates,
                    last_enqueued=last_enqueued,
                    now=12.1,
                )
                rows = outbox.connection.execute(
                    "SELECT outbox_id, payload_json FROM shelf_position_outbox ORDER BY outbox_id"
                ).fetchall()
                self.assertEqual(len(rows), 2)
                selected = json.loads(rows[0]["payload_json"])
                cleared = json.loads(rows[1]["payload_json"])
                self.assertEqual(selected["shelfId"], 2)
                self.assertEqual(selected["distanceMm"], 500)
                self.assertEqual(selected["sourceRevision"], 1)
                self.assertIsNone(cleared["shelfId"])
                self.assertIsNone(cleared["distanceMm"])
                self.assertEqual(cleared["sourceRevision"], 2)
            finally:
                outbox.close()

    def test_failed_delivery_is_retried_from_persistent_outbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempts = []

            def sender(payload):
                attempts.append(payload)
                if len(attempts) == 1:
                    raise RuntimeError("temporary failure")

            path = Path(temporary) / "state.sqlite"
            outbox = ShelfPositionOutbox(path)
            service = ShelfPositionPushService(
                db_path=path,
                shop_id=1,
                sender=sender,
            )
            try:
                persisted = outbox.enqueue(
                    {
                        "shopId": 1,
                        "visitId": 4,
                        "customerId": "customer-4",
                        "shelfId": 2,
                        "distanceMm": 500,
                        "observedAt": "2026-08-15T12:00:00Z",
                    },
                    0,
                )
                service._update_status(
                    4,
                    local={"shelfId": 2, "distanceMm": 500},
                    pendingCount=1,
                )
                service._deliver_one(outbox)
                row = outbox.connection.execute(
                    "SELECT status, attempts FROM shelf_position_outbox WHERE outbox_id=?",
                    (persisted["sourceRevision"],),
                ).fetchone()
                self.assertEqual((row["status"], row["attempts"]), ("pending", 1))
                with outbox.connection:
                    outbox.connection.execute(
                        "UPDATE shelf_position_outbox SET next_attempt_unix_ms=0"
                    )
                service._deliver_one(outbox)
                row = outbox.connection.execute(
                    "SELECT status, attempts FROM shelf_position_outbox WHERE outbox_id=?",
                    (persisted["sourceRevision"],),
                ).fetchone()
                self.assertEqual((row["status"], row["attempts"]), ("delivered", 1))
                self.assertEqual(len(attempts), 2)
                status = service.status_payload(4)
                self.assertEqual(status["status"], "synced")
                self.assertEqual(status["cloud"]["shelfId"], 2)
            finally:
                outbox.close()

    def test_reconciliation_reports_cloud_mismatch(self) -> None:
        service = ShelfPositionPushService(
            db_path=Path("unused.sqlite"),
            shop_id=1,
            sender=lambda _payload: None,
            reader=lambda visit_id: {
                "shopId": 1,
                "visitId": visit_id,
                "shelfId": 5,
                "distanceMm": 900,
                "sourceRevision": 12,
            },
        )
        service._update_status(
            4,
            local={"shelfId": 3, "distanceMm": 500},
            pendingCount=0,
        )

        output = io.StringIO()
        schedule = {}
        with contextlib.redirect_stdout(output):
            service._reconcile_one(schedule, 1.0)
            service._reconcile_one(schedule, 7.0)

        status = service.status_payload(4)
        self.assertEqual(status["status"], "mismatch")
        self.assertEqual(status["local"]["shelfId"], 3)
        self.assertEqual(status["cloud"]["shelfId"], 5)
        diagnostics = output.getvalue()
        self.assertEqual(diagnostics.count("SHOP_API_SHELF_RECONCILED"), 1)
        self.assertIn("status=mismatch", diagnostics)
        self.assertIn("local_shelf_id=3", diagnostics)
        self.assertIn("cloud_shelf_id=5", diagnostics)


if __name__ == "__main__":
    unittest.main()
