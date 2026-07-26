from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pipeline.shelf_api import shelf_event_payload
from pipeline.shelf_proximity import ShelfProximityEvent


DEFAULT_SHOP_STATE_DB = Path("state") / "shop_state.sqlite"


class ShopStateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def next_visit_id(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(visit_id), 0) + 1 AS next_id FROM visits").fetchone()
        return int(row["next_id"])

    def load_shop_customer_bindings(self) -> dict[int, str]:
        rows = self.connection.execute(
            """
            SELECT visit_id, shopping_customer_id
            FROM visits
            WHERE shopping_customer_id IS NOT NULL AND shopping_customer_id <> ''
            """
        ).fetchall()
        return {int(row["visit_id"]): str(row["shopping_customer_id"]) for row in rows}

    def record_entry(
        self,
        *,
        visit_id: int | None,
        host_seconds: float,
        device_id: str,
        camera_role: str,
        track_id: int,
        depth_mm: float | None,
        plane_signed_distance_mm: float | None,
        reason: str | None,
        event_payload: dict[str, Any],
    ) -> None:
        if visit_id is None:
            return
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO visits (
                    visit_id,
                    status,
                    origin,
                    first_entry_host_seconds,
                    last_entry_host_seconds,
                    last_seen_host_seconds,
                    last_device_id,
                    last_track_id,
                    updated_host_seconds
                )
                VALUES (?, 'inside', 'entrance_confirmed', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(visit_id) DO UPDATE SET
                    status='inside',
                    origin='entrance_confirmed',
                    first_entry_host_seconds=COALESCE(visits.first_entry_host_seconds, excluded.first_entry_host_seconds),
                    last_entry_host_seconds=excluded.last_entry_host_seconds,
                    last_seen_host_seconds=excluded.last_seen_host_seconds,
                    last_device_id=excluded.last_device_id,
                    last_track_id=excluded.last_track_id,
                    updated_host_seconds=excluded.updated_host_seconds
                """,
                (
                    visit_id,
                    host_seconds,
                    host_seconds,
                    host_seconds,
                    device_id,
                    track_id,
                    host_seconds,
                ),
            )
            self._insert_event(
                event_type="entry",
                visit_id=visit_id,
                host_seconds=host_seconds,
                device_id=device_id,
                camera_role=camera_role,
                track_id=track_id,
                depth_mm=depth_mm,
                plane_signed_distance_mm=plane_signed_distance_mm,
                reason=reason,
                payload=event_payload,
            )

    def record_leave(
        self,
        *,
        visit_id: int | None,
        host_seconds: float,
        device_id: str,
        camera_role: str,
        track_id: int,
        depth_mm: float | None,
        plane_signed_distance_mm: float | None,
        reason: str | None,
        event_payload: dict[str, Any],
    ) -> None:
        if visit_id is None:
            return
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO visits (
                    visit_id,
                    status,
                    last_leave_host_seconds,
                    last_seen_host_seconds,
                    last_device_id,
                    last_track_id,
                    updated_host_seconds
                )
                VALUES (?, 'left', ?, ?, ?, ?, ?)
                ON CONFLICT(visit_id) DO UPDATE SET
                    status='left',
                    last_leave_host_seconds=excluded.last_leave_host_seconds,
                    last_seen_host_seconds=excluded.last_seen_host_seconds,
                    last_device_id=excluded.last_device_id,
                    last_track_id=excluded.last_track_id,
                    updated_host_seconds=excluded.updated_host_seconds
                """,
                (
                    visit_id,
                    host_seconds,
                    host_seconds,
                    device_id,
                    track_id,
                    host_seconds,
                ),
            )
            self._insert_event(
                event_type="leave",
                visit_id=visit_id,
                host_seconds=host_seconds,
                device_id=device_id,
                camera_role=camera_role,
                track_id=track_id,
                depth_mm=depth_mm,
                plane_signed_distance_mm=plane_signed_distance_mm,
                reason=reason,
                payload=event_payload,
            )

    def record_shop_customer_binding(self, *, visit_id: int | None, shopping_customer_id: str | None) -> None:
        if visit_id is None or not shopping_customer_id:
            return
        with self.connection:
            self.connection.execute(
                """
                UPDATE visits
                SET shopping_customer_id = ?, updated_host_seconds = COALESCE(updated_host_seconds, 0)
                WHERE visit_id = ?
                """,
                (shopping_customer_id, visit_id),
            )

    def record_shelf_event(
        self,
        event: ShelfProximityEvent,
    ) -> ShelfProximityEvent:
        payload = shelf_event_payload(event)
        with self.connection:
            if event.event_type == "shelf_approach":
                self.connection.execute(
                    """
                    INSERT INTO shelf_proximity_sessions (
                        proximity_session_id,
                        shelf_id,
                        marker_id,
                        visit_id,
                        shopping_customer_id,
                        status,
                        approached_host_seconds,
                        approached_at_unix_ms,
                        minimum_distance_mm,
                        last_distance_mm,
                        last_device_id,
                        last_camera_index,
                        last_track_id,
                        updated_at_unix_ms
                    )
                    VALUES (?, ?, ?, ?, ?, 'near', ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(proximity_session_id) DO UPDATE SET
                        shopping_customer_id=COALESCE(
                            excluded.shopping_customer_id,
                            shelf_proximity_sessions.shopping_customer_id
                        ),
                        status='near',
                        minimum_distance_mm=MIN(
                            shelf_proximity_sessions.minimum_distance_mm,
                            excluded.minimum_distance_mm
                        ),
                        last_distance_mm=excluded.last_distance_mm,
                        last_device_id=excluded.last_device_id,
                        last_camera_index=excluded.last_camera_index,
                        last_track_id=excluded.last_track_id,
                        updated_at_unix_ms=excluded.updated_at_unix_ms
                    """,
                    (
                        event.proximity_session_id,
                        event.shelf_id,
                        event.marker_id,
                        event.visit_id,
                        event.customer_id,
                        event.host_synced_seconds,
                        event.occurred_at_unix_milliseconds,
                        event.distance_mm,
                        event.distance_mm,
                        event.device_id,
                        event.camera_index,
                        event.track_id,
                        event.occurred_at_unix_milliseconds,
                    ),
                )
            elif event.event_type == "shelf_departure":
                self.connection.execute(
                    """
                    UPDATE shelf_proximity_sessions
                    SET
                        status='departed',
                        shopping_customer_id=COALESCE(
                            ?,
                            shopping_customer_id
                        ),
                        departed_host_seconds=?,
                        departed_at_unix_ms=?,
                        minimum_distance_mm=MIN(
                            minimum_distance_mm,
                            ?
                        ),
                        last_distance_mm=?,
                        last_device_id=?,
                        last_camera_index=?,
                        last_track_id=?,
                        updated_at_unix_ms=?
                    WHERE proximity_session_id=?
                    """,
                    (
                        event.customer_id,
                        event.host_synced_seconds,
                        event.occurred_at_unix_milliseconds,
                        event.distance_mm,
                        event.distance_mm,
                        event.device_id,
                        event.camera_index,
                        event.track_id,
                        event.occurred_at_unix_milliseconds,
                        event.proximity_session_id,
                    ),
                )
            else:
                raise ValueError(f"Unsupported shelf event type: {event.event_type}")

            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO shelf_events (
                    event_type,
                    proximity_session_id,
                    shelf_id,
                    marker_id,
                    visit_id,
                    shopping_customer_id,
                    host_seconds,
                    occurred_at_unix_ms,
                    device_id,
                    camera_index,
                    track_id,
                    distance_mm,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_type,
                    event.proximity_session_id,
                    event.shelf_id,
                    event.marker_id,
                    event.visit_id,
                    event.customer_id,
                    event.host_synced_seconds,
                    event.occurred_at_unix_milliseconds,
                    event.device_id,
                    event.camera_index,
                    event.track_id,
                    event.distance_mm,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            if cursor.rowcount:
                event_id = int(cursor.lastrowid)
            else:
                row = self.connection.execute(
                    """
                    SELECT event_id
                    FROM shelf_events
                    WHERE proximity_session_id=? AND event_type=?
                    """,
                    (event.proximity_session_id, event.event_type),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Could not resolve persisted shelf event id.")
                event_id = int(row["event_id"])
            persisted_event = event.with_event_id(event_id)
            persisted_payload = shelf_event_payload(persisted_event)
            self.connection.execute(
                "UPDATE shelf_events SET payload_json=? WHERE event_id=?",
                (json.dumps(persisted_payload, sort_keys=True), event_id),
            )
        return persisted_event

    def load_recent_shelf_event_payloads(
        self,
        *,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("Shelf event limit must be positive.")
        rows = self.connection.execute(
            """
            SELECT event_id, payload_json
            FROM shelf_events
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in reversed(rows):
            payload = json.loads(str(row["payload_json"]))
            payload["eventId"] = int(row["event_id"])
            payloads.append(payload)
        return payloads

    def load_active_shelf_session_payloads(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                sessions.proximity_session_id,
                sessions.minimum_distance_mm,
                events.payload_json
            FROM shelf_proximity_sessions AS sessions
            JOIN shelf_events AS events
              ON events.proximity_session_id = sessions.proximity_session_id
             AND events.event_type = 'shelf_approach'
            WHERE sessions.status = 'near'
            ORDER BY sessions.updated_at_unix_ms
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            payload["minimumDistanceMm"] = (
                None
                if row["minimum_distance_mm"] is None
                else float(row["minimum_distance_mm"])
            )
            result.append(payload)
        return result

    def _insert_event(
        self,
        *,
        event_type: str,
        visit_id: int,
        host_seconds: float,
        device_id: str,
        camera_role: str,
        track_id: int,
        depth_mm: float | None,
        plane_signed_distance_mm: float | None,
        reason: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO visit_events (
                event_type,
                visit_id,
                host_seconds,
                device_id,
                camera_role,
                track_id,
                depth_mm,
                plane_signed_distance_mm,
                reason,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                visit_id,
                host_seconds,
                device_id,
                camera_role,
                track_id,
                depth_mm,
                plane_signed_distance_mm,
                reason,
                json.dumps(payload, sort_keys=True),
            ),
        )

    def _initialize(self) -> None:
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    visit_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    origin TEXT,
                    shopping_customer_id TEXT,
                    first_entry_host_seconds REAL,
                    last_entry_host_seconds REAL,
                    last_leave_host_seconds REAL,
                    last_seen_host_seconds REAL,
                    last_device_id TEXT,
                    last_track_id INTEGER,
                    updated_host_seconds REAL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS visit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    visit_id INTEGER NOT NULL,
                    host_seconds REAL NOT NULL,
                    device_id TEXT NOT NULL,
                    camera_role TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    depth_mm REAL,
                    plane_signed_distance_mm REAL,
                    reason TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_visit_events_visit_id ON visit_events(visit_id)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_visit_events_host_seconds ON visit_events(host_seconds)"
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS shelf_proximity_sessions (
                    proximity_session_id TEXT PRIMARY KEY,
                    shelf_id INTEGER NOT NULL,
                    marker_id INTEGER NOT NULL,
                    visit_id INTEGER NOT NULL,
                    shopping_customer_id TEXT,
                    status TEXT NOT NULL,
                    approached_host_seconds REAL,
                    approached_at_unix_ms INTEGER,
                    departed_host_seconds REAL,
                    departed_at_unix_ms INTEGER,
                    minimum_distance_mm REAL,
                    last_distance_mm REAL,
                    last_device_id TEXT,
                    last_camera_index INTEGER,
                    last_track_id INTEGER,
                    updated_at_unix_ms INTEGER NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS shelf_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    proximity_session_id TEXT NOT NULL,
                    shelf_id INTEGER NOT NULL,
                    marker_id INTEGER NOT NULL,
                    visit_id INTEGER NOT NULL,
                    shopping_customer_id TEXT,
                    host_seconds REAL NOT NULL,
                    occurred_at_unix_ms INTEGER NOT NULL,
                    device_id TEXT NOT NULL,
                    camera_index INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    distance_mm REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(proximity_session_id, event_type)
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shelf_events_shelf_time
                ON shelf_events(shelf_id, occurred_at_unix_ms)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shelf_events_visit_time
                ON shelf_events(visit_id, occurred_at_unix_ms)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shelf_events_session
                ON shelf_events(proximity_session_id)
                """
            )
