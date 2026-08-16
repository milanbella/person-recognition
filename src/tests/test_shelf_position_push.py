import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_position_push import (
    MAX_DELIVERY_ATTEMPTS,
    ShelfPositionOutbox,
    ShelfPositionPushService,
    _retry_delay_seconds,
)
from pipeline.shelf_proximity import ShelfCameraObservation


def _observation(
    *,
    shelf_id: int,
    distance_mm: float,
    customer_id: str | None = "customer-4",
) -> ShelfCameraObservation:
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
        customer_id=customer_id,
        distance_mm=distance_mm,
        person_point_3d_mm=(0.0, 0.0, 2000.0),
        anchor=anchor,
        host_synced_seconds=1.0,
        observed_at_unix_milliseconds=1000,
        rgb_sequence_number=12,
        depth_sequence_number=12,
    )


class ShelfPositionPushTests(unittest.TestCase):
    def test_retry_backoff_is_capped_at_eight_seconds(self) -> None:
        self.assertEqual(
            [_retry_delay_seconds(attempt) for attempt in range(1, 8)],
            [1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0],
        )

    def test_terminal_history_is_bounded_without_deleting_active_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
            try:
                with outbox.connection:
                    for visit_id, status in (
                        (4, "delivered"),
                        (4, "superseded"),
                        (4, "failed"),
                        (5, "failed"),
                        (6, "pending"),
                        (7, "sending"),
                    ):
                        outbox.connection.execute(
                            """
                            INSERT INTO shelf_position_outbox (
                                visit_id, payload_json, status,
                                next_attempt_unix_ms
                            ) VALUES (?, ?, ?, 0)
                            """,
                            (
                                visit_id,
                                json.dumps(
                                    {
                                        "visitId": visit_id,
                                        "customerId": f"customer-{visit_id}",
                                        "shelfId": 2,
                                    }
                                ),
                                status,
                            ),
                        )

                self.assertEqual(outbox.prune_terminal_rows(retain=1), 2)
                rows = outbox.connection.execute(
                    """
                    SELECT visit_id, status
                    FROM shelf_position_outbox
                    ORDER BY outbox_id
                    """
                ).fetchall()
                self.assertEqual(
                    [(row["visit_id"], row["status"]) for row in rows],
                    [
                        (4, "failed"),
                        (5, "failed"),
                        (6, "pending"),
                        (7, "sending"),
                    ],
                )
            finally:
                outbox.close()

    def test_delivery_stops_after_final_eight_second_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.sqlite"
            attempts = []

            def sender(_payload):
                attempts.append(1)
                raise RuntimeError("server unavailable")

            outbox = ShelfPositionOutbox(path)
            service = ShelfPositionPushService(
                db_path=path,
                shop_id=1,
                sender=sender,
                stability_seconds=0.0,
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
                service._update_status(4, pendingCount=1)
                for _attempt in range(MAX_DELIVERY_ATTEMPTS):
                    service._deliver_one(outbox)
                    with outbox.connection:
                        outbox.connection.execute(
                            """
                            UPDATE shelf_position_outbox
                            SET next_attempt_unix_ms=0
                            WHERE status='pending'
                            """
                        )

                row = outbox.connection.execute(
                    """
                    SELECT status, attempts, last_error
                    FROM shelf_position_outbox
                    WHERE outbox_id=?
                    """,
                    (persisted["sourceRevision"],),
                ).fetchone()
                self.assertEqual(len(attempts), MAX_DELIVERY_ATTEMPTS)
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["attempts"], MAX_DELIVERY_ATTEMPTS)
                self.assertEqual(row["last_error"], "server unavailable")
                self.assertIsNone(outbox.next_due(2**63 - 1))
                status = service.status_payload(4)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["pendingCount"], 0)

                observations = {
                    (0, 2, 4, 12): (
                        _observation(shelf_id=2, distance_mm=500),
                        20.0,
                    )
                }
                service._evaluate(
                    outbox=outbox,
                    observations=observations,
                    candidates={},
                    last_enqueued={4: (2, 0.0)},
                    now=20.0,
                )
                row_count = outbox.connection.execute(
                    "SELECT COUNT(*) FROM shelf_position_outbox"
                ).fetchone()[0]
                self.assertEqual(row_count, 1)
            finally:
                outbox.close()

    def test_new_position_supersedes_older_pending_position_for_visit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
            try:
                first = outbox.enqueue(
                    {
                        "shopId": 1,
                        "visitId": 4,
                        "customerId": "customer-4",
                        "shelfId": 4,
                        "distanceMm": 500,
                        "observedAt": "2026-08-15T12:00:00Z",
                    },
                    0,
                )
                second = outbox.enqueue(
                    {
                        "shopId": 1,
                        "visitId": 4,
                        "customerId": "customer-4",
                        "shelfId": 3,
                        "distanceMm": 450,
                        "observedAt": "2026-08-15T12:00:01Z",
                    },
                    0,
                )

                rows = outbox.connection.execute(
                    "SELECT outbox_id, status FROM shelf_position_outbox ORDER BY outbox_id"
                ).fetchall()
                self.assertEqual(
                    [(row["outbox_id"], row["status"]) for row in rows],
                    [
                        (first["sourceRevision"], "superseded"),
                        (second["sourceRevision"], "pending"),
                    ],
                )
                due = outbox.next_due(0)
                self.assertEqual(due["outbox_id"], second["sourceRevision"])
                self.assertEqual(json.loads(due["payload_json"])["shelfId"], 3)
                self.assertEqual(outbox.pending_count(4), 1)
            finally:
                outbox.close()

    def test_startup_compaction_keeps_latest_pending_position_per_visit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
            try:
                with outbox.connection:
                    for shelf_id in (4, 3, 2):
                        outbox.connection.execute(
                            """
                            INSERT INTO shelf_position_outbox (
                                visit_id, payload_json, next_attempt_unix_ms
                            ) VALUES (?, ?, 0)
                            """,
                            (
                                4,
                                json.dumps(
                                    {
                                        "visitId": 4,
                                        "customerId": "customer-4",
                                        "shelfId": shelf_id,
                                    }
                                ),
                            ),
                        )
                    outbox.connection.execute(
                        """
                        INSERT INTO shelf_position_outbox (
                            visit_id, payload_json, next_attempt_unix_ms
                        ) VALUES (5, ?, 0)
                        """,
                        (
                            json.dumps(
                                {
                                    "visitId": 5,
                                    "customerId": "customer-5",
                                    "shelfId": 1,
                                }
                            ),
                        ),
                    )

                self.assertEqual(outbox.coalesce_pending(), 2)
                rows = outbox.pending_rows()
                self.assertEqual(len(rows), 2)
                self.assertEqual(
                    [(row["visit_id"], json.loads(row["payload_json"])["shelfId"]) for row in rows],
                    [(4, 2), (5, 1)],
                )
            finally:
                outbox.close()

    def test_interrupted_delivery_is_recovered_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
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
                self.assertTrue(outbox.mark_sending(persisted["sourceRevision"]))
                self.assertIsNone(outbox.next_due(0))

                self.assertEqual(outbox.recover_interrupted_sends(10), 1)
                self.assertIsNone(outbox.next_due(9))
                self.assertEqual(
                    outbox.next_due(10)["outbox_id"],
                    persisted["sourceRevision"],
                )
            finally:
                outbox.close()

    def test_failed_in_flight_position_yields_to_newer_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.sqlite"
            outbox = ShelfPositionOutbox(path)
            sent_shelf_ids = []

            def sender(payload):
                sent_shelf_ids.append(payload["shelfId"])
                if len(sent_shelf_ids) == 1:
                    outbox.enqueue(
                        {
                            "shopId": 1,
                            "visitId": 4,
                            "customerId": "customer-4",
                            "shelfId": 3,
                            "distanceMm": 450,
                            "observedAt": "2026-08-15T12:00:01Z",
                        },
                        0,
                    )
                    raise RuntimeError("temporary failure")
                return payload

            service = ShelfPositionPushService(
                db_path=path,
                shop_id=1,
                sender=sender,
            )
            try:
                first = outbox.enqueue(
                    {
                        "shopId": 1,
                        "visitId": 4,
                        "customerId": "customer-4",
                        "shelfId": 4,
                        "distanceMm": 500,
                        "observedAt": "2026-08-15T12:00:00Z",
                    },
                    0,
                )
                service._update_status(4, pendingCount=1)

                service._deliver_one(outbox)
                first_row = outbox.connection.execute(
                    "SELECT status FROM shelf_position_outbox WHERE outbox_id=?",
                    (first["sourceRevision"],),
                ).fetchone()
                self.assertEqual(first_row["status"], "superseded")
                self.assertEqual(outbox.pending_count(4), 1)

                service._deliver_one(outbox)
                self.assertEqual(sent_shelf_ids, [4, 3])
                self.assertEqual(outbox.pending_count(4), 0)
                self.assertEqual(service.status_payload(4)["cloud"]["shelfId"], 3)
            finally:
                outbox.close()

    def test_outbox_rejects_unbound_shelf_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
            try:
                with self.assertRaisesRegex(ValueError, "bound customerId"):
                    outbox.enqueue(
                        {
                            "shopId": 1,
                            "visitId": 4,
                            "customerId": None,
                            "shelfId": 2,
                            "distanceMm": 500,
                            "observedAt": "2026-08-15T12:00:00Z",
                        },
                        0,
                    )
                count = outbox.connection.execute(
                    "SELECT COUNT(*) FROM shelf_position_outbox"
                ).fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                outbox.close()

    def test_legacy_pending_unbound_positions_are_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outbox = ShelfPositionOutbox(Path(temporary) / "state.sqlite")
            try:
                with outbox.connection:
                    outbox.connection.execute(
                        """
                        INSERT INTO shelf_position_outbox (
                            visit_id, payload_json, next_attempt_unix_ms
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            4,
                            json.dumps(
                                {
                                    "shopId": 1,
                                    "visitId": 4,
                                    "customerId": None,
                                    "shelfId": 2,
                                }
                            ),
                            0,
                        ),
                    )

                self.assertEqual(outbox.discard_pending_unbound(), 1)
                row = outbox.connection.execute(
                    "SELECT status, last_error FROM shelf_position_outbox"
                ).fetchone()
                self.assertEqual(row["status"], "discarded")
                self.assertEqual(row["last_error"], "visit_not_bound_to_customer")
                self.assertIsNone(outbox.next_due(0))
            finally:
                outbox.close()

    def test_unbound_observation_is_not_enqueued(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.sqlite"
            outbox = ShelfPositionOutbox(path)
            service = ShelfPositionPushService(
                db_path=path,
                shop_id=1,
                sender=lambda _payload: None,
                stability_seconds=0.0,
            )
            try:
                service._evaluate(
                    outbox=outbox,
                    observations={
                        (0, 2, 4, 12): (
                            _observation(
                                shelf_id=2,
                                distance_mm=500,
                                customer_id=None,
                            ),
                            10.0,
                        ),
                    },
                    candidates={},
                    last_enqueued={},
                    now=10.0,
                )
                count = outbox.connection.execute(
                    "SELECT COUNT(*) FROM shelf_position_outbox"
                ).fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                outbox.close()

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
                    "SELECT outbox_id, payload_json, status FROM shelf_position_outbox ORDER BY outbox_id"
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
                self.assertEqual(rows[0]["status"], "superseded")
                self.assertEqual(rows[1]["status"], "pending")
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
