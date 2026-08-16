from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pipeline.shelf_api import ShelfCameraSnapshot
from pipeline.shelf_proximity import ShelfCameraObservation


MAX_DELIVERY_ATTEMPTS = 5
TERMINAL_ROW_RETENTION = 1000
OUTBOX_PRUNE_INTERVAL_SECONDS = 60.0
TERMINAL_STATUSES = ("delivered", "superseded", "discarded", "failed")


def _retry_delay_seconds(attempts: int) -> float:
    return min(8.0, float(2 ** min(attempts - 1, 3)))


@dataclass
class _Candidate:
    shelf_id: int | None
    since_monotonic: float
    distance_mm: int | None = None
    customer_id: str | None = None
    observed_at_unix_ms: int | None = None


class ShelfPositionOutbox:
    def __init__(self, db_path: Path) -> None:
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS shelf_position_outbox (
                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_unix_ms INTEGER NOT NULL,
                    last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TEXT
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shelf_position_outbox_pending
                ON shelf_position_outbox(status, next_attempt_unix_ms, outbox_id)
                """
            )

    def close(self) -> None:
        self.connection.close()

    def enqueue(self, payload: dict[str, object], now_ms: int) -> dict[str, object]:
        customer_id = payload.get("customerId")
        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError("Shelf position payload requires a bound customerId.")
        visit_id = int(payload["visitId"])
        with self.connection:
            self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='superseded', last_error='superseded_by_newer_position'
                WHERE visit_id=? AND status='pending'
                """,
                (visit_id,),
            )
            cursor = self.connection.execute(
                """
                INSERT INTO shelf_position_outbox (
                    visit_id, payload_json, next_attempt_unix_ms
                ) VALUES (?, '{}', ?)
                """,
                (visit_id, now_ms),
            )
            revision = int(cursor.lastrowid)
            persisted = {**payload, "sourceRevision": revision}
            self.connection.execute(
                "UPDATE shelf_position_outbox SET payload_json=? WHERE outbox_id=?",
                (json.dumps(persisted, sort_keys=True), revision),
            )
        return persisted

    def next_due(self, now_ms: int) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT outbox_id, visit_id, payload_json, attempts
            FROM shelf_position_outbox
            WHERE status='pending' AND next_attempt_unix_ms <= ?
            ORDER BY outbox_id
            LIMIT 1
            """,
            (now_ms,),
        ).fetchone()

    def pending_rows(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT visit_id, payload_json, attempts, last_error
            FROM shelf_position_outbox
            WHERE status='pending'
            ORDER BY outbox_id
            """
        ).fetchall()

    def latest_failed_rows(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT visit_id, payload_json, attempts, last_error
            FROM shelf_position_outbox
            WHERE status='failed'
              AND outbox_id IN (
                  SELECT MAX(outbox_id)
                  FROM shelf_position_outbox
                  WHERE status='failed'
                  GROUP BY visit_id
              )
            ORDER BY outbox_id
            """
        ).fetchall()

    def prune_terminal_rows(self, retain: int = TERMINAL_ROW_RETENTION) -> int:
        if retain < 0:
            raise ValueError("Terminal outbox retention must not be negative.")
        placeholders = ", ".join("?" for _status in TERMINAL_STATUSES)
        with self.connection:
            cursor = self.connection.execute(
                f"""
                DELETE FROM shelf_position_outbox
                WHERE status IN ({placeholders})
                  AND outbox_id NOT IN (
                      SELECT MAX(outbox_id)
                      FROM shelf_position_outbox
                      WHERE status IN ({placeholders})
                      GROUP BY visit_id
                  )
                  AND outbox_id NOT IN (
                      SELECT outbox_id
                      FROM shelf_position_outbox
                      WHERE status IN ({placeholders})
                      ORDER BY outbox_id DESC
                      LIMIT ?
                  )
                """,
                (
                    *TERMINAL_STATUSES,
                    *TERMINAL_STATUSES,
                    *TERMINAL_STATUSES,
                    retain,
                ),
            )
        return cursor.rowcount

    def mark_delivered(self, outbox_id: int) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='delivered', delivered_at=CURRENT_TIMESTAMP, last_error=NULL
                WHERE outbox_id=?
                """,
                (outbox_id,),
            )

    def mark_sending(self, outbox_id: int) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='sending'
                WHERE outbox_id=? AND status='pending'
                """,
                (outbox_id,),
            )
        return cursor.rowcount == 1

    def recover_interrupted_sends(self, now_ms: int) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='pending', next_attempt_unix_ms=?,
                    last_error='delivery_interrupted'
                WHERE status='sending'
                """,
                (now_ms,),
            )
        return cursor.rowcount

    def coalesce_pending(self) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='superseded', last_error='superseded_by_newer_position'
                WHERE status='pending'
                  AND outbox_id NOT IN (
                      SELECT MAX(outbox_id)
                      FROM shelf_position_outbox
                      WHERE status='pending'
                      GROUP BY visit_id
                  )
                """
            )
        return cursor.rowcount

    def has_newer_pending(self, visit_id: int, outbox_id: int) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM shelf_position_outbox
            WHERE visit_id=? AND status='pending' AND outbox_id>?
            LIMIT 1
            """,
            (visit_id, outbox_id),
        ).fetchone()
        return row is not None

    def pending_count(self, visit_id: int) -> int:
        return int(
            self.connection.execute(
                """
                SELECT COUNT(*)
                FROM shelf_position_outbox
                WHERE visit_id=? AND status IN ('pending', 'sending')
                """,
                (visit_id,),
            ).fetchone()[0]
        )

    def mark_superseded(self, outbox_id: int, reason: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='superseded', last_error=?
                WHERE outbox_id=?
                """,
                (reason[:1000], outbox_id),
            )

    def mark_failed(self, outbox_id: int, *, attempts: int, error: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='failed', attempts=?, last_error=?
                WHERE outbox_id=?
                """,
                (attempts, error[:1000], outbox_id),
            )

    def discard_pending_unbound(self) -> int:
        rows = self.connection.execute(
            """
            SELECT outbox_id, payload_json
            FROM shelf_position_outbox
            WHERE status='pending'
            """
        ).fetchall()
        discarded_ids = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            customer_id = payload.get("customerId")
            if not isinstance(customer_id, str) or not customer_id.strip():
                discarded_ids.append(int(row["outbox_id"]))
        if not discarded_ids:
            return 0
        with self.connection:
            self.connection.executemany(
                """
                UPDATE shelf_position_outbox
                SET status='discarded', last_error='visit_not_bound_to_customer'
                WHERE outbox_id=?
                """,
                ((outbox_id,) for outbox_id in discarded_ids),
            )
        return len(discarded_ids)

    def mark_retry(
        self,
        outbox_id: int,
        *,
        attempts: int,
        next_attempt_unix_ms: int,
        error: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE shelf_position_outbox
                SET status='pending', attempts=?, next_attempt_unix_ms=?, last_error=?
                WHERE outbox_id=?
                """,
                (attempts, next_attempt_unix_ms, error[:1000], outbox_id),
            )


class ShelfPositionPushService:
    """Debounces nearest-shelf state and delivers it through a durable outbox."""

    def __init__(
        self,
        *,
        db_path: Path,
        shop_id: int,
        sender: Callable[[dict[str, object]], object],
        reader: Callable[[int], dict[str, object] | None] | None = None,
        stability_seconds: float = 0.75,
        stale_seconds: float = 2.0,
        heartbeat_seconds: float = 15.0,
        reconcile_seconds: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self.shop_id = shop_id
        self.sender = sender
        self.reader = reader
        self.stability_seconds = stability_seconds
        self.stale_seconds = stale_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.reconcile_seconds = reconcile_seconds
        self._snapshots: queue.SimpleQueue[ShelfCameraSnapshot] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._status_by_visit: dict[int, dict[str, object]] = {}
        self._reconcile_log_signatures: dict[int, tuple[object, ...]] = {}
        self._terminal_failed_shelves: dict[int, int | None] = {}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Shelf position push service is already running.")
        self._thread = threading.Thread(
            target=self._run,
            name="shelf-position-push",
            daemon=True,
        )
        self._thread.start()

    def publish(self, snapshot: ShelfCameraSnapshot) -> None:
        self._snapshots.put(snapshot)

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
            self._thread = None

    def status_payload(self, visit_id: int) -> dict[str, object]:
        now_ms = time.time_ns() // 1_000_000
        with self._status_lock:
            values = dict(self._status_by_visit.get(visit_id, {}))
        if not values:
            return {
                "enabled": True,
                "visitId": visit_id,
                "status": "unknown",
                "local": None,
                "cloud": None,
                "pendingCount": 0,
                "lastError": None,
            }
        local = values.get("local")
        cloud = values.get("cloud")
        pending_count = int(values.get("pendingCount", 0))
        last_error = values.get("lastError")
        debounce_until = values.get("debounceUntilUnixMilliseconds")
        debounce_remaining = (
            None
            if debounce_until is None
            else max(0, int(debounce_until) - now_ms)
        )
        local_shelf = None if not isinstance(local, dict) else local.get("shelfId")
        cloud_shelf = None if not isinstance(cloud, dict) else cloud.get("shelfId")
        if values.get("deliveryFailed"):
            status = "failed"
        elif pending_count and last_error:
            status = "retrying"
        elif pending_count:
            status = "queued"
        elif debounce_remaining:
            status = "pending"
        elif values.get("reconcileError"):
            status = "unavailable"
        elif cloud is None:
            status = "unknown"
        elif local_shelf == cloud_shelf:
            status = "cleared" if local_shelf is None else "synced"
        else:
            status = "mismatch"
        return {
            "enabled": True,
            "visitId": visit_id,
            "status": status,
            "local": local,
            "cloud": cloud,
            "pendingCount": pending_count,
            "attempts": int(values.get("attempts", 0)),
            "debounceRemainingMilliseconds": debounce_remaining,
            "lastSuccessfulSyncUnixMilliseconds": values.get(
                "lastSuccessfulSyncUnixMilliseconds"
            ),
            "lastReconciledAtUnixMilliseconds": values.get(
                "lastReconciledAtUnixMilliseconds"
            ),
            "lastError": last_error or values.get("reconcileError"),
        }

    def _update_status(self, visit_id: int, **values: object) -> None:
        with self._status_lock:
            self._status_by_visit.setdefault(visit_id, {}).update(values)

    def _log_reconciliation(
        self,
        visit_id: int,
        *,
        reconcile_error: str | None,
    ) -> None:
        payload = self.status_payload(visit_id)
        local = payload.get("local")
        cloud = payload.get("cloud")
        local_values = local if isinstance(local, dict) else {}
        cloud_values = cloud if isinstance(cloud, dict) else {}
        signature = (
            payload.get("status"),
            local_values.get("shelfId"),
            cloud_values.get("shelfId"),
            cloud_values.get("sourceRevision"),
            payload.get("pendingCount"),
            reconcile_error,
        )
        if self._reconcile_log_signatures.get(visit_id) == signature:
            return
        self._reconcile_log_signatures[visit_id] = signature
        print(
            f"SHOP_API_SHELF_RECONCILED visit_id={visit_id} "
            f"status={payload.get('status')} "
            f"local_shelf_id={local_values.get('shelfId')} "
            f"local_distance_mm={local_values.get('distanceMm')} "
            f"cloud_shelf_id={cloud_values.get('shelfId')} "
            f"cloud_distance_mm={cloud_values.get('distanceMm')} "
            f"cloud_revision={cloud_values.get('sourceRevision')} "
            f"pending_count={payload.get('pendingCount')} "
            f"error={reconcile_error}"
        )

    def _run(self) -> None:
        outbox = ShelfPositionOutbox(self.db_path)
        observations: dict[
            tuple[int, int, int, int], tuple[ShelfCameraObservation, float]
        ] = {}
        candidates: dict[int, _Candidate] = {}
        last_enqueued: dict[int, tuple[int | None, float]] = {}
        next_reconcile: dict[int, float] = {}
        next_prune = time.monotonic() + OUTBOX_PRUNE_INTERVAL_SECONDS
        try:
            recovered_count = outbox.recover_interrupted_sends(
                time.time_ns() // 1_000_000
            )
            if recovered_count:
                print(
                    "SHOP_API_SHELF_RECOVERED_IN_FLIGHT "
                    f"count={recovered_count}"
                )
            discarded_count = outbox.discard_pending_unbound()
            if discarded_count:
                print(
                    "SHOP_API_SHELF_DISCARDED_UNBOUND "
                    f"count={discarded_count}"
                )
            coalesced_count = outbox.coalesce_pending()
            if coalesced_count:
                print(f"SHOP_API_SHELF_COALESCED count={coalesced_count}")
            pruned_count = outbox.prune_terminal_rows()
            if pruned_count:
                print(f"SHOP_API_SHELF_OUTBOX_PRUNED count={pruned_count}")
            for row in outbox.latest_failed_rows():
                visit_id = int(row["visit_id"])
                payload = json.loads(str(row["payload_json"]))
                shelf_value = payload.get("shelfId")
                shelf_id = None if shelf_value is None else int(shelf_value)
                self._terminal_failed_shelves[visit_id] = shelf_id
                last_enqueued[visit_id] = (shelf_id, time.monotonic())
                self._update_status(
                    visit_id,
                    local={
                        "shelfId": shelf_id,
                        "distanceMm": payload.get("distanceMm"),
                        "observedAt": payload.get("observedAt"),
                    },
                    pendingCount=0,
                    attempts=int(row["attempts"]),
                    lastError=row["last_error"],
                    deliveryFailed=True,
                )
            pending_by_visit: dict[int, dict[str, object]] = {}
            for row in outbox.pending_rows():
                visit_id = int(row["visit_id"])
                payload = json.loads(str(row["payload_json"]))
                pending = pending_by_visit.setdefault(
                    visit_id,
                    {
                        "count": 0,
                        "attempts": 0,
                        "lastError": None,
                        "payload": payload,
                    },
                )
                pending["count"] = int(pending["count"]) + 1
                pending["attempts"] = max(
                    int(pending["attempts"]), int(row["attempts"])
                )
                if row["last_error"]:
                    pending["lastError"] = row["last_error"]
                    pending["payload"] = payload
            for visit_id, pending in pending_by_visit.items():
                payload = pending["payload"]
                assert isinstance(payload, dict)
                self._terminal_failed_shelves.pop(visit_id, None)
                self._update_status(
                    visit_id,
                    local={
                        "shelfId": payload.get("shelfId"),
                        "distanceMm": payload.get("distanceMm"),
                        "observedAt": payload.get("observedAt"),
                    },
                    pendingCount=int(pending["count"]),
                    attempts=int(pending["attempts"]),
                    lastError=pending["lastError"],
                    deliveryFailed=False,
                )
            while not self._stop.wait(0.1):
                now = time.monotonic()
                while True:
                    try:
                        snapshot = self._snapshots.get_nowait()
                    except queue.Empty:
                        break
                    for observation in snapshot.observations:
                        if observation.visit_id is None or not observation.customer_id:
                            continue
                        observations[
                            (
                                snapshot.camera_index,
                                observation.shelf_id,
                                int(observation.visit_id),
                                observation.marker_id,
                            )
                        ] = (observation, now)
                self._evaluate(
                    outbox=outbox,
                    observations=observations,
                    candidates=candidates,
                    last_enqueued=last_enqueued,
                    now=now,
                )
                self._deliver_one(outbox)
                self._reconcile_one(next_reconcile, now)
                if now >= next_prune:
                    pruned_count = outbox.prune_terminal_rows()
                    if pruned_count:
                        print(
                            "SHOP_API_SHELF_OUTBOX_PRUNED "
                            f"count={pruned_count}"
                        )
                    next_prune = now + OUTBOX_PRUNE_INTERVAL_SECONDS
        finally:
            outbox.close()

    def _evaluate(
        self,
        *,
        outbox: ShelfPositionOutbox,
        observations: dict[
            tuple[int, int, int, int], tuple[ShelfCameraObservation, float]
        ],
        candidates: dict[int, _Candidate],
        last_enqueued: dict[int, tuple[int | None, float]],
        now: float,
    ) -> None:
        for key, (_observation, received_at) in list(observations.items()):
            if now - received_at > self.stale_seconds:
                observations.pop(key, None)

        closest: dict[int, ShelfCameraObservation] = {}
        for observation, _received_at in observations.values():
            assert observation.visit_id is not None
            if not observation.customer_id:
                continue
            visit_id = int(observation.visit_id)
            current = closest.get(visit_id)
            if current is None or observation.distance_mm < current.distance_mm:
                closest[visit_id] = observation

        visit_ids = set(closest) | set(last_enqueued) | set(candidates)
        for visit_id in visit_ids:
            observation = closest.get(visit_id)
            shelf_id = None if observation is None else int(observation.shelf_id)
            candidate = candidates.get(visit_id)
            if candidate is None or candidate.shelf_id != shelf_id:
                candidate = _Candidate(
                    shelf_id=shelf_id,
                    since_monotonic=now,
                    customer_id=(None if candidate is None else candidate.customer_id),
                )
                candidates[visit_id] = candidate
            if observation is not None:
                candidate.distance_mm = int(round(observation.distance_mm))
                candidate.customer_id = observation.customer_id
                candidate.observed_at_unix_ms = observation.observed_at_unix_milliseconds
            else:
                candidate.distance_mm = None
                candidate.observed_at_unix_ms = None

            debounce_remaining = max(
                0.0,
                self.stability_seconds - (now - candidate.since_monotonic),
            )
            self._update_status(
                visit_id,
                local={
                    "shelfId": candidate.shelf_id,
                    "distanceMm": candidate.distance_mm,
                    "observedAtUnixMilliseconds": candidate.observed_at_unix_ms,
                },
                debounceUntilUnixMilliseconds=(
                    time.time_ns() // 1_000_000
                    + int(round(debounce_remaining * 1000))
                ),
            )

            if now - candidate.since_monotonic < self.stability_seconds:
                continue
            previous = last_enqueued.get(visit_id)
            changed = previous is None or previous[0] != candidate.shelf_id
            failed_same_state = (
                visit_id in self._terminal_failed_shelves
                and self._terminal_failed_shelves[visit_id] == candidate.shelf_id
            )
            if changed and not failed_same_state:
                self._terminal_failed_shelves.pop(visit_id, None)
            heartbeat_due = (
                candidate.shelf_id is not None
                and previous is not None
                and now - previous[1] >= self.heartbeat_seconds
                and not failed_same_state
            )
            if failed_same_state:
                changed = False
            if not changed and not heartbeat_due:
                continue
            if candidate.shelf_id is None and previous is None:
                continue

            observed_ms = candidate.observed_at_unix_ms or time.time_ns() // 1_000_000
            observed_at = datetime.fromtimestamp(
                observed_ms / 1000.0,
                tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            payload: dict[str, object] = {
                "shopId": self.shop_id,
                "visitId": visit_id,
                "customerId": candidate.customer_id,
                "shelfId": candidate.shelf_id,
                "distanceMm": candidate.distance_mm,
                "observedAt": observed_at,
            }
            persisted = outbox.enqueue(payload, time.time_ns() // 1_000_000)
            last_enqueued[visit_id] = (candidate.shelf_id, now)
            self._update_status(
                visit_id,
                pendingCount=outbox.pending_count(visit_id),
                attempts=0,
                lastError=None,
                deliveryFailed=False,
                debounceUntilUnixMilliseconds=None,
            )
            print(
                f"SHOP_API_SHELF_QUEUED visit_id={visit_id} "
                f"shelf_id={candidate.shelf_id} distance_mm={candidate.distance_mm} "
                f"source_revision={persisted['sourceRevision']}"
            )

    def _deliver_one(self, outbox: ShelfPositionOutbox) -> None:
        now_ms = time.time_ns() // 1_000_000
        row = outbox.next_due(now_ms)
        if row is None:
            return
        outbox_id = int(row["outbox_id"])
        visit_id = int(row["visit_id"])
        payload = json.loads(str(row["payload_json"]))
        if not outbox.mark_sending(outbox_id):
            return
        try:
            response = self.sender(payload)
        except Exception as exc:
            if outbox.has_newer_pending(visit_id, outbox_id):
                outbox.mark_superseded(
                    outbox_id,
                    "superseded_after_failed_delivery",
                )
                self._update_status(
                    visit_id,
                    pendingCount=outbox.pending_count(visit_id),
                    attempts=0,
                    lastError=None,
                )
                print(
                    f"SHOP_API_SHELF_SUPERSEDED source_revision={outbox_id} "
                    f"visit_id={visit_id} error={exc}"
                )
                return
            attempts = int(row["attempts"]) + 1
            if attempts >= MAX_DELIVERY_ATTEMPTS:
                outbox.mark_failed(
                    outbox_id,
                    attempts=attempts,
                    error=str(exc),
                )
                shelf_value = payload.get("shelfId")
                self._terminal_failed_shelves[visit_id] = (
                    None if shelf_value is None else int(shelf_value)
                )
                self._update_status(
                    visit_id,
                    pendingCount=outbox.pending_count(visit_id),
                    attempts=attempts,
                    lastError=str(exc),
                    deliveryFailed=True,
                )
                print(
                    f"SHOP_API_SHELF_FAILED source_revision={outbox_id} "
                    f"visit_id={visit_id} attempts={attempts} error={exc}"
                )
                return
            delay_seconds = _retry_delay_seconds(attempts)
            outbox.mark_retry(
                outbox_id,
                attempts=attempts,
                next_attempt_unix_ms=now_ms + int(delay_seconds * 1000),
                error=str(exc),
            )
            self._update_status(
                visit_id,
                pendingCount=outbox.pending_count(visit_id),
                attempts=attempts,
                lastError=str(exc),
                deliveryFailed=False,
            )
            print(
                f"SHOP_API_SHELF_RETRY source_revision={outbox_id} "
                f"attempt={attempts} delay_seconds={delay_seconds:.1f} error={exc}"
            )
            return
        outbox.mark_delivered(outbox_id)
        self._terminal_failed_shelves.pop(visit_id, None)
        cloud = response if isinstance(response, dict) else payload
        self._update_status(
            visit_id,
            cloud=dict(cloud),
            pendingCount=outbox.pending_count(visit_id),
            attempts=0,
            lastError=None,
            deliveryFailed=False,
            reconcileError=None,
            lastSuccessfulSyncUnixMilliseconds=time.time_ns() // 1_000_000,
        )
        print(
            f"SHOP_API_SHELF_DELIVERED visit_id={payload['visitId']} "
            f"shelf_id={payload.get('shelfId')} source_revision={outbox_id}"
        )

    def _reconcile_one(
        self,
        next_reconcile: dict[int, float],
        now: float,
    ) -> None:
        if self.reader is None:
            return
        with self._status_lock:
            visit_ids = sorted(self._status_by_visit)
        visit_id = next(
            (
                item
                for item in visit_ids
                if now >= next_reconcile.get(item, 0.0)
            ),
            None,
        )
        if visit_id is None:
            return
        next_reconcile[visit_id] = now + self.reconcile_seconds
        try:
            cloud = self.reader(visit_id)
        except Exception as exc:
            self._update_status(
                visit_id,
                reconcileError=str(exc),
                lastReconciledAtUnixMilliseconds=time.time_ns() // 1_000_000,
            )
            self._log_reconciliation(visit_id, reconcile_error=str(exc))
            return
        self._update_status(
            visit_id,
            cloud=None if cloud is None else dict(cloud),
            reconcileError=None,
            lastReconciledAtUnixMilliseconds=time.time_ns() // 1_000_000,
        )
        self._log_reconciliation(visit_id, reconcile_error=None)
