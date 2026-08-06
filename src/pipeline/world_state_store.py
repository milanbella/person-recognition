from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


class WorldStateStore:
    """Coalesced SQLite persistence for the live system-belief projection."""

    REVISION_BLOCK_SIZE = 1_000_000

    def __init__(self, db_path: Path, *, flush_interval_seconds: float = 0.35) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval_seconds = flush_interval_seconds
        self.process_instance_id = str(uuid.uuid4())
        self._initialize()
        self._condition = threading.Condition()
        self._pending_snapshot: dict[str, Any] | None = None
        self._pending_changes: deque[dict[str, Any]] = deque()
        self._writer_busy = False
        self._closed = False
        self._writer_error: BaseException | None = None
        self._query_lock = threading.Lock()
        self._query_cache: dict[str, dict[str, Any]] = {}
        self._query_cache_order: deque[str] = deque()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="world-state-writer",
            daemon=True,
        )
        self._writer.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS world_state_meta (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    schema_version INTEGER NOT NULL,
                    current_revision INTEGER NOT NULL,
                    allocated_revision_ceiling INTEGER NOT NULL,
                    process_instance_id TEXT NOT NULL,
                    updated_at_unix_ms INTEGER NOT NULL,
                    source_event_id_high_watermark INTEGER
                );

                CREATE TABLE IF NOT EXISTS world_state_cameras (
                    camera_index INTEGER PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    observed_at_unix_ms INTEGER,
                    freshness TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS world_state_visits (
                    visit_id INTEGER PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    status TEXT,
                    origin TEXT,
                    customer_id TEXT,
                    observed_at_unix_ms INTEGER,
                    freshness TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS world_state_shelves (
                    shelf_id INTEGER PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    observed_at_unix_ms INTEGER,
                    freshness TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS world_state_changes (
                    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision INTEGER NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    occurred_at_unix_ms INTEGER NOT NULL,
                    host_synced_seconds REAL,
                    source_event_id INTEGER,
                    source_json TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_world_changes_revision
                ON world_state_changes(revision);

                CREATE TABLE IF NOT EXISTS world_state_query_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    process_instance_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    run_id TEXT,
                    subject_id TEXT,
                    generated_at_unix_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_world_query_revision
                ON world_state_query_snapshots(revision, generated_at_unix_ms);
                """
            )
            now_ms = time.time_ns() // 1_000_000
            connection.execute(
                """
                INSERT OR IGNORE INTO world_state_meta (
                    singleton_id, schema_version, current_revision,
                    allocated_revision_ceiling, process_instance_id,
                    updated_at_unix_ms
                ) VALUES (1, 1, 0, 0, ?, ?)
                """,
                (self.process_instance_id, now_ms),
            )

    def reserve_revision_block(self) -> tuple[int, int]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT allocated_revision_ceiling FROM world_state_meta WHERE singleton_id=1"
            ).fetchone()
            previous_ceiling = int(row["allocated_revision_ceiling"])
            start = previous_ceiling + 1
            ceiling = previous_ceiling + self.REVISION_BLOCK_SIZE
            connection.execute(
                """
                UPDATE world_state_meta
                SET allocated_revision_ceiling=?, process_instance_id=?,
                    updated_at_unix_ms=?
                WHERE singleton_id=1
                """,
                (ceiling, self.process_instance_id, time.time_ns() // 1_000_000),
            )
            connection.commit()
        return start, ceiling

    def load_entities(self) -> dict[str, list[dict[str, Any]]]:
        with self._connection() as connection:
            result: dict[str, list[dict[str, Any]]] = {}
            for name, table in (
                ("cameras", "world_state_cameras"),
                ("visits", "world_state_visits"),
                ("shelves", "world_state_shelves"),
            ):
                rows = connection.execute(f"SELECT state_json FROM {table}").fetchall()
                result[name] = [json.loads(str(row["state_json"])) for row in rows]
            existing_visits = {
                int(item["visitId"]): item for item in result["visits"]
            }
            if self._table_exists(connection, "visits"):
                rows = connection.execute(
                    """
                    SELECT visit_id, status, origin, shopping_customer_id,
                           last_seen_host_seconds, last_device_id, last_track_id,
                           updated_host_seconds
                    FROM visits
                    """
                ).fetchall()
                for row in rows:
                    visit_id = int(row["visit_id"])
                    visit = existing_visits.setdefault(
                        visit_id,
                        {"visitId": visit_id, "currentTracks": []},
                    )
                    visit.update(
                        {
                            "status": str(row["status"]),
                            "origin": row["origin"],
                            "customerId": row["shopping_customer_id"],
                            "lastSeenHostSeconds": row["last_seen_host_seconds"],
                            "lastDeviceId": row["last_device_id"],
                            "lastTrackId": row["last_track_id"],
                            "updatedHostSeconds": row["updated_host_seconds"],
                        }
                    )
            result["visits"] = list(existing_visits.values())

            existing_shelves = {
                int(item["shelfId"]): item for item in result["shelves"]
            }
            if self._table_exists(connection, "shelf_proximity_sessions"):
                rows = connection.execute(
                    """
                    SELECT shelf_id, marker_id, visit_id, shopping_customer_id,
                           proximity_session_id, last_distance_mm,
                           last_camera_index, last_device_id, last_track_id,
                           updated_at_unix_ms
                    FROM shelf_proximity_sessions
                    WHERE status='near'
                    """
                ).fetchall()
                for row in rows:
                    shelf_id = int(row["shelf_id"])
                    shelf = existing_shelves.setdefault(
                        shelf_id,
                        {"shelfId": shelf_id, "markerIds": [int(row["marker_id"])]},
                    )
                    shelf.update(
                        {
                            "state": "occupied",
                            "ownerVisitId": int(row["visit_id"]),
                            "ownerCustomerId": row["shopping_customer_id"],
                            "ownerTrackId": row["last_track_id"],
                            "sourceCameraIndex": row["last_camera_index"],
                            "sourceDeviceId": row["last_device_id"],
                            "distanceMm": row["last_distance_mm"],
                            "proximitySessionId": row["proximity_session_id"],
                            "observedAtUnixMilliseconds": row["updated_at_unix_ms"],
                        }
                    )
            result["shelves"] = list(existing_shelves.values())
        return result

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def enqueue(
        self,
        snapshot: Mapping[str, Any],
        *,
        changes: list[Mapping[str, Any]] | None = None,
    ) -> None:
        with self._condition:
            if self._closed:
                return
            self._pending_snapshot = dict(snapshot)
            for change in changes or ():
                self._pending_changes.append(dict(change))
            self._condition.notify()

    def flush(self) -> None:
        deadline = time.monotonic() + 5.0
        with self._condition:
            self._condition.notify()
            while (
                (
                    self._pending_snapshot is not None
                    or self._pending_changes
                    or self._writer_busy
                )
                and self._writer_error is None
                and time.monotonic() < deadline
            ):
                self._condition.wait(timeout=0.05)
        if self._writer_error is not None:
            raise RuntimeError("World-state persistence failed.") from self._writer_error

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._writer.join(timeout=5.0)
        if self._writer.is_alive():
            raise RuntimeError("World-state persistence worker did not stop.")
        if self._writer_error is not None:
            raise RuntimeError("World-state persistence failed.") from self._writer_error

    def capture_query_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        kind: str,
        run_id: str | None = None,
        subject_id: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        captured = dict(payload)
        snapshot_id = str(uuid.uuid4())
        captured["snapshotId"] = snapshot_id
        now_ms = time.time_ns() // 1_000_000
        with self._query_lock:
            self._query_cache[snapshot_id] = captured
            self._query_cache_order.append(snapshot_id)
            while len(self._query_cache_order) > 512:
                expired = self._query_cache_order.popleft()
                self._query_cache.pop(expired, None)
        if not persist:
            return captured
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO world_state_query_snapshots (
                    snapshot_id, revision, process_instance_id, kind,
                    run_id, subject_id, generated_at_unix_ms, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    int(captured["revision"]),
                    str(captured["processInstanceId"]),
                    kind,
                    run_id,
                    subject_id,
                    now_ms,
                    json.dumps(captured, sort_keys=True),
                ),
            )
        return captured

    def query_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._query_lock:
            cached = self._query_cache.get(snapshot_id)
            if cached is not None:
                return dict(cached)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM world_state_query_snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        return None if row is None else json.loads(str(row["payload_json"]))

    def revision_snapshot(self, revision: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM world_state_query_snapshots
                WHERE revision=?
                ORDER BY generated_at_unix_ms DESC
                LIMIT 1
                """,
                (revision,),
            ).fetchone()
        return None if row is None else json.loads(str(row["payload_json"]))

    def _writer_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._closed
                        or self._pending_snapshot is not None
                        or bool(self._pending_changes),
                        timeout=self.flush_interval_seconds,
                    )
                    if (
                        self._closed
                        and self._pending_snapshot is None
                        and not self._pending_changes
                    ):
                        return
                    snapshot = self._pending_snapshot
                    changes = list(self._pending_changes)
                    self._pending_snapshot = None
                    self._pending_changes.clear()
                    self._writer_busy = True
                if snapshot is not None:
                    self._write_snapshot(snapshot, changes)
                elif changes:
                    self._write_changes(changes)
                with self._condition:
                    self._writer_busy = False
                    self._condition.notify_all()
        except BaseException as exc:
            self._writer_error = exc
            print(f"WORLD_STATE_PERSIST_ERROR error={exc}")
            with self._condition:
                self._writer_busy = False
                self._condition.notify_all()

    def _write_snapshot(
        self,
        snapshot: Mapping[str, Any],
        changes: list[Mapping[str, Any]],
    ) -> None:
        revision = int(snapshot["revision"])
        with self._connection() as connection:
            connection.execute("BEGIN")
            connection.execute(
                """
                UPDATE world_state_meta
                SET schema_version=?, current_revision=?, process_instance_id=?,
                    updated_at_unix_ms=?, source_event_id_high_watermark=?
                WHERE singleton_id=1
                """,
                (
                    int(snapshot["schemaVersion"]),
                    revision,
                    str(snapshot["processInstanceId"]),
                    int(snapshot["generatedAtUnixMilliseconds"]),
                    snapshot.get("sourceEventIdHighWatermark"),
                ),
            )
            for camera in snapshot.get("cameras", []):
                self._upsert_entity(
                    connection,
                    table="world_state_cameras",
                    id_column="camera_index",
                    entity_id=int(camera["cameraIndex"]),
                    revision=revision,
                    observed_at=camera.get("observedAtUnixMilliseconds"),
                    freshness=str(camera.get("freshness", "unknown")),
                    payload=camera,
                )
            for visit in snapshot.get("visits", []):
                connection.execute(
                    """
                    INSERT INTO world_state_visits (
                        visit_id, revision, status, origin, customer_id,
                        observed_at_unix_ms, freshness, state_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(visit_id) DO UPDATE SET
                        revision=excluded.revision,
                        status=excluded.status,
                        origin=excluded.origin,
                        customer_id=excluded.customer_id,
                        observed_at_unix_ms=excluded.observed_at_unix_ms,
                        freshness=excluded.freshness,
                        state_json=excluded.state_json
                    """,
                    (
                        int(visit["visitId"]),
                        revision,
                        visit.get("status"),
                        visit.get("origin"),
                        visit.get("customerId"),
                        visit.get("observedAtUnixMilliseconds"),
                        str(visit.get("freshness", "unknown")),
                        json.dumps(visit, sort_keys=True),
                    ),
                )
            for shelf in snapshot.get("shelves", []):
                self._upsert_entity(
                    connection,
                    table="world_state_shelves",
                    id_column="shelf_id",
                    entity_id=int(shelf["shelfId"]),
                    revision=revision,
                    observed_at=shelf.get("observedAtUnixMilliseconds"),
                    freshness=str(shelf.get("freshness", "unknown")),
                    payload=shelf,
                )
            self._insert_changes(connection, changes)
            connection.commit()

    def _write_changes(self, changes: list[Mapping[str, Any]]) -> None:
        with self._connection() as connection:
            self._insert_changes(connection, changes)
            connection.commit()

    @staticmethod
    def _upsert_entity(
        connection: sqlite3.Connection,
        *,
        table: str,
        id_column: str,
        entity_id: int,
        revision: int,
        observed_at: Any,
        freshness: str,
        payload: Mapping[str, Any],
    ) -> None:
        connection.execute(
            f"""
            INSERT INTO {table} (
                {id_column}, revision, observed_at_unix_ms, freshness, state_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT({id_column}) DO UPDATE SET
                revision=excluded.revision,
                observed_at_unix_ms=excluded.observed_at_unix_ms,
                freshness=excluded.freshness,
                state_json=excluded.state_json
            """,
            (
                entity_id,
                revision,
                observed_at,
                freshness,
                json.dumps(payload, sort_keys=True),
            ),
        )

    @staticmethod
    def _insert_changes(
        connection: sqlite3.Connection,
        changes: list[Mapping[str, Any]],
    ) -> None:
        for change in changes:
            connection.execute(
                """
                INSERT INTO world_state_changes (
                    revision, entity_type, entity_id, change_type,
                    occurred_at_unix_ms, host_synced_seconds, source_event_id,
                    source_json, before_json, after_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(change["revision"]),
                    str(change["entityType"]),
                    str(change["entityId"]),
                    str(change["changeType"]),
                    int(change["occurredAtUnixMilliseconds"]),
                    change.get("hostSyncedSeconds"),
                    change.get("sourceEventId"),
                    json.dumps(change.get("source", {}), sort_keys=True),
                    None
                    if change.get("before") is None
                    else json.dumps(change["before"], sort_keys=True),
                    None
                    if change.get("after") is None
                    else json.dumps(change["after"], sort_keys=True),
                ),
            )
