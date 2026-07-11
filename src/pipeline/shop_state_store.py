from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


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
