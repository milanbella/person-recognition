from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SESSION_SCENARIOS = {
    "clear",
    "hand",
    "shelf",
    "walking",
    "moderate_occlusion",
    "heavy_occlusion",
    "pocket",
    "other",
}
FRAME_OUTCOMES = {"accepted", "corrected", "not_visible", "rejected", "uncertain"}


class ModelTrainingStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

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
                CREATE TABLE IF NOT EXISTS mt_products (
                    product_code TEXT PRIMARY KEY,
                    source_product_id INTEGER,
                    shop_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    barcode TEXT,
                    active INTEGER NOT NULL,
                    refreshed_at_unix_ms INTEGER NOT NULL,
                    source_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mt_capture_sessions (
                    session_id TEXT PRIMARY KEY,
                    shop_id INTEGER NOT NULL,
                    product_code TEXT NOT NULL,
                    product_name_snapshot TEXT NOT NULL,
                    source_product_id_snapshot INTEGER,
                    scenario TEXT NOT NULL,
                    camera_index INTEGER NOT NULL,
                    device_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset_intent TEXT NOT NULL,
                    started_at_unix_ms INTEGER NOT NULL,
                    stopped_at_unix_ms INTEGER,
                    operator_active_ms INTEGER NOT NULL DEFAULT 0,
                    notes TEXT,
                    FOREIGN KEY(product_code) REFERENCES mt_products(product_code)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_mt_one_active_session
                ON mt_capture_sessions(status) WHERE status = 'active';

                CREATE TABLE IF NOT EXISTS mt_captured_frames (
                    frame_id TEXT PRIMARY KEY,
                    capture_request_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    camera_index INTEGER NOT NULL,
                    device_id TEXT NOT NULL,
                    rgb_sequence_number INTEGER,
                    captured_at_unix_ms INTEGER NOT NULL,
                    image_path TEXT NOT NULL UNIQUE,
                    metadata_path TEXT NOT NULL,
                    source_capture_path TEXT,
                    review_proxy_path TEXT NOT NULL,
                    thumbnail_path TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    perceptual_hash TEXT NOT NULL,
                    annotation_status TEXT NOT NULL,
                    review_outcome TEXT,
                    created_at_unix_ms INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES mt_capture_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_mt_frames_review
                ON mt_captured_frames(annotation_status, captured_at_unix_ms);

                CREATE TABLE IF NOT EXISTS mt_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    frame_id TEXT NOT NULL,
                    product_code TEXT NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    coordinate_space TEXT NOT NULL,
                    source TEXT NOT NULL,
                    provider_name TEXT,
                    provider_version TEXT,
                    provider_run_id TEXT,
                    provider_config_json TEXT,
                    confidence REAL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at_unix_ms INTEGER NOT NULL,
                    updated_at_unix_ms INTEGER NOT NULL,
                    FOREIGN KEY(frame_id) REFERENCES mt_captured_frames(frame_id),
                    FOREIGN KEY(product_code) REFERENCES mt_products(product_code)
                );

                CREATE TABLE IF NOT EXISTS mt_annotation_revisions (
                    revision_id TEXT PRIMARY KEY,
                    frame_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    annotations_json TEXT NOT NULL,
                    created_at_unix_ms INTEGER NOT NULL,
                    FOREIGN KEY(frame_id) REFERENCES mt_captured_frames(frame_id),
                    UNIQUE(frame_id, revision)
                );

                CREATE TABLE IF NOT EXISTS mt_dataset_versions (
                    dataset_version TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at_unix_ms INTEGER NOT NULL
                );
                """
            )

    def replace_products(self, shop_id: int, products: Sequence[Mapping[str, Any]]) -> int:
        now_ms = _now_ms()
        normalized: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        for product in products:
            code = str(product.get("code") or "").strip()
            name = str(product.get("name") or product.get("label") or "").strip()
            if not code or not name:
                raise ValueError("Every product requires non-empty code and name.")
            if code in seen:
                raise ValueError(f"Duplicate product code: {code}")
            seen.add(code)
            source_id = product.get("id")
            normalized.append(
                (
                    code,
                    None if source_id is None else int(source_id),
                    shop_id,
                    name,
                    None if product.get("barCode") is None else str(product["barCode"]),
                    1 if bool(product.get("active", True)) else 0,
                    now_ms,
                    json.dumps(dict(product), sort_keys=True),
                )
            )
        with self._connection() as connection:
            connection.execute("UPDATE mt_products SET active = 0 WHERE shop_id = ?", (shop_id,))
            connection.executemany(
                """
                INSERT INTO mt_products (
                    product_code, source_product_id, shop_id, name, barcode,
                    active, refreshed_at_unix_ms, source_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_code) DO UPDATE SET
                    source_product_id = excluded.source_product_id,
                    shop_id = excluded.shop_id,
                    name = excluded.name,
                    barcode = excluded.barcode,
                    active = excluded.active,
                    refreshed_at_unix_ms = excluded.refreshed_at_unix_ms,
                    source_json = excluded.source_json
                """,
                normalized,
            )
        return len(normalized)

    def list_products(self, *, active_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM mt_products"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY name COLLATE NOCASE, product_code"
        with self._connection() as connection:
            rows = connection.execute(sql).fetchall()
        return [_product_payload(row) for row in rows]

    def create_session(
        self,
        *,
        shop_id: int,
        product_code: str,
        scenario: str,
        camera_index: int,
        device_id: str,
        dataset_intent: str = "development",
        notes: str | None = None,
    ) -> dict[str, Any]:
        if scenario not in SESSION_SCENARIOS:
            raise ValueError(f"Unsupported scenario: {scenario}")
        if dataset_intent not in {"development", "gold_test"}:
            raise ValueError("datasetIntent must be development or gold_test.")
        if camera_index < 0:
            raise ValueError("cameraIndex must not be negative.")
        session_id = str(uuid.uuid4())
        with self._connection() as connection:
            product = connection.execute(
                "SELECT * FROM mt_products WHERE product_code = ? AND active = 1",
                (product_code,),
            ).fetchone()
            if product is None:
                raise ValueError(f"Unknown active product code: {product_code}")
            connection.execute("UPDATE mt_capture_sessions SET status = 'stopped', stopped_at_unix_ms = ? WHERE status = 'active'", (_now_ms(),))
            connection.execute(
                """
                INSERT INTO mt_capture_sessions (
                    session_id, shop_id, product_code, product_name_snapshot,
                    source_product_id_snapshot, scenario, camera_index,
                    device_id, status, dataset_intent, started_at_unix_ms, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    session_id,
                    shop_id,
                    product_code,
                    product["name"],
                    product["source_product_id"],
                    scenario,
                    camera_index,
                    device_id,
                    dataset_intent,
                    _now_ms(),
                    notes,
                ),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT s.*, COUNT(f.frame_id) AS frame_count
                FROM mt_capture_sessions s
                LEFT JOIN mt_captured_frames f ON f.session_id = s.session_id
                WHERE s.session_id = ? GROUP BY s.session_id
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _session_payload(row)

    def active_session(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT session_id FROM mt_capture_sessions WHERE status = 'active'"
            ).fetchone()
        return None if row is None else self.get_session(str(row["session_id"]))

    def stop_session(self, session_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE mt_capture_sessions SET status = 'stopped', stopped_at_unix_ms = ? WHERE session_id = ? AND status = 'active'",
                (_now_ms(), session_id),
            )
            if cursor.rowcount == 0 and connection.execute(
                "SELECT 1 FROM mt_capture_sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise KeyError(session_id)
        return self.get_session(session_id)

    def add_frame(self, record: Mapping[str, Any]) -> dict[str, Any]:
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT frame_id FROM mt_captured_frames WHERE capture_request_id = ?",
                (record["capture_request_id"],),
            ).fetchone()
            if existing is not None:
                return self.get_frame(str(existing["frame_id"]))
            connection.execute(
                """
                INSERT INTO mt_captured_frames (
                    frame_id, capture_request_id, session_id, camera_index,
                    device_id, rgb_sequence_number, captured_at_unix_ms,
                    image_path, metadata_path, source_capture_path,
                    review_proxy_path, thumbnail_path, width, height, sha256,
                    perceptual_hash, annotation_status, created_at_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'needs_review', ?)
                """,
                (
                    record["frame_id"], record["capture_request_id"],
                    record["session_id"], record["camera_index"], record["device_id"],
                    record.get("rgb_sequence_number"), record["captured_at_unix_ms"],
                    record["image_path"], record["metadata_path"],
                    record.get("source_capture_path"), record["review_proxy_path"],
                    record["thumbnail_path"], record["width"], record["height"],
                    record["sha256"], record["perceptual_hash"], _now_ms(),
                ),
            )
        return self.get_frame(str(record["frame_id"]))

    def get_frame(self, frame_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT f.*, s.product_code, s.product_name_snapshot, s.scenario,
                       s.dataset_intent
                FROM mt_captured_frames f
                JOIN mt_capture_sessions s ON s.session_id = f.session_id
                WHERE f.frame_id = ?
                """,
                (frame_id,),
            ).fetchone()
            if row is None:
                raise KeyError(frame_id)
            annotations = connection.execute(
                "SELECT * FROM mt_annotations WHERE frame_id = ? AND status = 'accepted' ORDER BY annotation_id",
                (frame_id,),
            ).fetchall()
        payload = _frame_payload(row)
        payload["annotations"] = [_annotation_payload(item) for item in annotations]
        return payload

    def list_frames(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if status:
            where = "WHERE f.annotation_status = ?"
            params.append(status)
        params.append(max(1, min(limit, 500)))
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT f.frame_id FROM mt_captured_frames f {where}
                ORDER BY f.captured_at_unix_ms ASC LIMIT ?
                """,
                params,
            ).fetchall()
        return [self.get_frame(str(row["frame_id"])) for row in rows]

    def frame_image_path(self, frame_id: str, variant: str) -> Path:
        columns = {
            "original": "image_path",
            "review": "review_proxy_path",
            "thumbnail": "thumbnail_path",
        }
        column = columns.get(variant)
        if column is None:
            raise ValueError("variant must be original, review, or thumbnail.")
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {column} AS path FROM mt_captured_frames WHERE frame_id = ?",
                (frame_id,),
            ).fetchone()
        if row is None:
            raise KeyError(frame_id)
        return Path(str(row["path"]))

    def save_annotations(
        self,
        frame_id: str,
        boxes: Sequence[Mapping[str, Any]],
        *,
        actor: str = "operator",
    ) -> dict[str, Any]:
        frame = self.get_frame(frame_id)
        normalized = [_validate_box(box, str(frame["productCode"])) for box in boxes]
        now_ms = _now_ms()
        with self._connection() as connection:
            revision = int(connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM mt_annotation_revisions WHERE frame_id = ?",
                (frame_id,),
            ).fetchone()[0])
            connection.execute("DELETE FROM mt_annotations WHERE frame_id = ?", (frame_id,))
            for box in normalized:
                connection.execute(
                    """
                    INSERT INTO mt_annotations (
                        annotation_id, frame_id, product_code, x1, y1, x2, y2,
                        coordinate_space, source, status, revision,
                        created_at_unix_ms, updated_at_unix_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'normalized_xyxy', 'manual',
                              'accepted', ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()), frame_id, box["productCode"], box["x1"],
                        box["y1"], box["x2"], box["y2"], revision, now_ms, now_ms,
                    ),
                )
            connection.execute(
                "UPDATE mt_captured_frames SET annotation_status = 'needs_review', review_outcome = NULL WHERE frame_id = ?",
                (frame_id,),
            )
            self._insert_revision(connection, frame_id, revision, "save_draft", actor, normalized)
        return self.get_frame(frame_id)

    def finalize_frame(self, frame_id: str, outcome: str, *, actor: str = "operator") -> dict[str, Any]:
        if outcome not in FRAME_OUTCOMES:
            raise ValueError(f"Unsupported frame outcome: {outcome}")
        frame = self.get_frame(frame_id)
        annotations = list(frame["annotations"])
        if outcome in {"accepted", "corrected"} and not annotations:
            raise ValueError("Accepted product frames require at least one box.")
        if outcome in {"not_visible", "rejected", "uncertain"}:
            annotations = []
        with self._connection() as connection:
            if not annotations:
                connection.execute("DELETE FROM mt_annotations WHERE frame_id = ?", (frame_id,))
            revision = int(connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM mt_annotation_revisions WHERE frame_id = ?",
                (frame_id,),
            ).fetchone()[0])
            connection.execute(
                "UPDATE mt_captured_frames SET annotation_status = ?, review_outcome = ? WHERE frame_id = ?",
                (outcome, outcome, frame_id),
            )
            self._insert_revision(connection, frame_id, revision, outcome, actor, annotations)
        return self.get_frame(frame_id)

    def export_rows(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT f.*, s.product_code, s.product_name_snapshot, s.scenario,
                       s.dataset_intent
                FROM mt_captured_frames f
                JOIN mt_capture_sessions s ON s.session_id = f.session_id
                WHERE f.annotation_status IN ('accepted', 'corrected', 'not_visible')
                ORDER BY f.session_id, f.captured_at_unix_ms
                """
            ).fetchall()
        return [self.get_frame(str(row["frame_id"])) for row in rows]

    def next_dataset_version(self) -> str:
        with self._connection() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM mt_dataset_versions").fetchone()[0])
        return f"dataset-v{count + 1:04d}"

    def save_dataset(self, version: str, path: Path, manifest: Mapping[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO mt_dataset_versions VALUES (?, ?, 'complete', ?, ?)",
                (version, str(path), json.dumps(dict(manifest), sort_keys=True), _now_ms()),
            )

    @staticmethod
    def _insert_revision(
        connection: sqlite3.Connection,
        frame_id: str,
        revision: int,
        action: str,
        actor: str,
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        connection.execute(
            "INSERT INTO mt_annotation_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()), frame_id, revision, action, actor,
                json.dumps(list(annotations), sort_keys=True), _now_ms(),
            ),
        )


def _validate_box(box: Mapping[str, Any], default_product_code: str) -> dict[str, Any]:
    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each box requires numeric x1, y1, x2, and y2.") from exc
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("Box coordinates must be normalized with x1 < x2 and y1 < y2.")
    return {
        "productCode": str(box.get("productCode") or default_product_code),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _product_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "code": row["product_code"], "id": row["source_product_id"],
        "shopId": row["shop_id"], "name": row["name"], "barCode": row["barcode"],
        "active": bool(row["active"]), "refreshedAtUnixMilliseconds": row["refreshed_at_unix_ms"],
    }


def _session_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sessionId": row["session_id"], "shopId": row["shop_id"],
        "productCode": row["product_code"], "productName": row["product_name_snapshot"],
        "scenario": row["scenario"], "cameraIndex": row["camera_index"],
        "cameraNumber": row["camera_index"] + 1, "deviceId": row["device_id"],
        "status": row["status"], "datasetIntent": row["dataset_intent"],
        "startedAtUnixMilliseconds": row["started_at_unix_ms"],
        "stoppedAtUnixMilliseconds": row["stopped_at_unix_ms"],
        "frameCount": int(row["frame_count"]) if "frame_count" in row.keys() else 0,
    }


def _frame_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "frameId": row["frame_id"], "sessionId": row["session_id"],
        "cameraIndex": row["camera_index"], "cameraNumber": row["camera_index"] + 1,
        "deviceId": row["device_id"], "rgbSequenceNumber": row["rgb_sequence_number"],
        "capturedAtUnixMilliseconds": row["captured_at_unix_ms"],
        "width": row["width"], "height": row["height"], "sha256": row["sha256"],
        "perceptualHash": row["perceptual_hash"],
        "annotationStatus": row["annotation_status"], "reviewOutcome": row["review_outcome"],
        "productCode": row["product_code"], "productName": row["product_name_snapshot"],
        "scenario": row["scenario"], "datasetIntent": row["dataset_intent"],
        "imageUrls": {
            "thumbnail": f"/model-training/api/frames/{row['frame_id']}/image?variant=thumbnail",
            "review": f"/model-training/api/frames/{row['frame_id']}/image?variant=review",
            "original": f"/model-training/api/frames/{row['frame_id']}/image?variant=original",
        },
    }


def _annotation_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "annotationId": row["annotation_id"], "productCode": row["product_code"],
        "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        "source": row["source"], "confidence": row["confidence"], "revision": row["revision"],
    }
