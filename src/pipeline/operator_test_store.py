from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pipeline.operator_models import (
    ANNOTATION_TYPES,
    OBSERVATION_ANNOTATION_TYPES,
    AnalysisResult,
    ObservationReference,
    OperatorEvent,
)


class OperatorTestStore:
    """SQLite-backed test runs using independent per-operation connections."""

    def __init__(
        self,
        db_path: Path,
        *,
        runs_root: Path = Path("test-runs"),
        runtime_configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.runs_root = Path(runs_root)
        self.runtime_configuration = dict(runtime_configuration or {})
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._event_queue: queue.Queue[OperatorEvent | None] = queue.Queue(
            maxsize=10000
        )
        self._writer_error: BaseException | None = None
        self._writer_closed = False
        self._writer_thread = threading.Thread(
            target=self._event_writer,
            name="operator-event-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operator_test_runs (
                    run_id TEXT PRIMARY KEY,
                    scenario TEXT NOT NULL,
                    verifier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at_unix_ms INTEGER NOT NULL,
                    stopped_at_unix_ms INTEGER,
                    git_commit TEXT,
                    configuration_json TEXT NOT NULL,
                    notes TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_operator_one_active_run
                ON operator_test_runs(status)
                WHERE status = 'active';

                CREATE TABLE IF NOT EXISTS operator_test_subjects (
                    run_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    expected_customer_id TEXT,
                    PRIMARY KEY (run_id, subject_id),
                    FOREIGN KEY (run_id) REFERENCES operator_test_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS operator_physical_visits (
                    physical_visit_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    entered_at_unix_ms INTEGER NOT NULL,
                    left_at_unix_ms INTEGER,
                    mapped_system_visit_id INTEGER,
                    mapping_status TEXT NOT NULL,
                    UNIQUE (run_id, subject_id, ordinal),
                    FOREIGN KEY (run_id, subject_id)
                        REFERENCES operator_test_subjects(run_id, subject_id)
                );

                CREATE TABLE IF NOT EXISTS operator_annotations (
                    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    annotation_type TEXT NOT NULL,
                    subject_id TEXT,
                    physical_visit_id TEXT,
                    received_at_unix_ms INTEGER NOT NULL,
                    client_recorded_at_unix_ms INTEGER,
                    observation_reference_json TEXT,
                    system_snapshot_json TEXT NOT NULL,
                    evidence_path TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES operator_test_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_operator_annotations_run_time
                ON operator_annotations(run_id, received_at_unix_ms);

                CREATE TABLE IF NOT EXISTS operator_system_events (
                    event_id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at_unix_ms INTEGER NOT NULL,
                    host_synced_seconds REAL,
                    camera_index INTEGER,
                    device_id TEXT,
                    track_id INTEGER,
                    visit_id INTEGER,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES operator_test_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_operator_events_run_time
                ON operator_system_events(run_id, occurred_at_unix_ms);

                CREATE TABLE IF NOT EXISTS operator_analysis_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expected_annotation_id INTEGER,
                    matched_event_id INTEGER,
                    latency_ms INTEGER,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    UNIQUE (run_id, rule_id, expected_annotation_id),
                    FOREIGN KEY (run_id) REFERENCES operator_test_runs(run_id)
                );
                """
            )

    def max_event_id(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(event_id), 0) AS value FROM operator_system_events"
            ).fetchone()
        return int(row["value"])

    def active_run(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM operator_test_runs
                WHERE status = 'active'
                ORDER BY started_at_unix_ms DESC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            return self._run_payload(connection, row)

    def start_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        scenario = str(payload.get("scenario", "")).strip()
        verifier = str(payload.get("verifier", "")).strip()
        if not scenario:
            raise ValueError("scenario is required.")
        if not verifier:
            raise ValueError("verifier is required.")
        subjects = payload.get("subjects")
        if not isinstance(subjects, list) or not subjects:
            raise ValueError("At least one test subject is required.")

        normalized_subjects: list[dict[str, str | None]] = []
        seen_subject_ids: set[str] = set()
        for item in subjects:
            if not isinstance(item, Mapping):
                raise ValueError("Each subject must be an object.")
            subject_id = self._slug(str(item.get("subjectId", "")).strip())
            display_name = str(item.get("displayName", "")).strip()
            if not subject_id or not display_name:
                raise ValueError("Each subject requires subjectId and displayName.")
            if subject_id in seen_subject_ids:
                raise ValueError(f"Duplicate subjectId: {subject_id}")
            seen_subject_ids.add(subject_id)
            customer_id = item.get("expectedCustomerId")
            normalized_subjects.append(
                {
                    "subjectId": subject_id,
                    "displayName": display_name,
                    "expectedCustomerId": (
                        None if customer_id in (None, "") else str(customer_id)
                    ),
                }
            )

        now_ms = time.time_ns() // 1_000_000
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now_ms / 1000))
        run_id = f"{stamp}-{self._slug(scenario)}-{uuid.uuid4().hex[:6]}"
        notes = payload.get("notes")
        git_commit = self.runtime_configuration.get("gitCommit")

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT run_id FROM operator_test_runs WHERE status='active'"
            ).fetchone()
            if active is not None:
                connection.rollback()
                raise RuntimeError(f"Test run {active['run_id']} is already active.")
            connection.execute(
                """
                INSERT INTO operator_test_runs (
                    run_id, scenario, verifier, status, started_at_unix_ms,
                    git_commit, configuration_json, notes
                )
                VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    scenario,
                    verifier,
                    now_ms,
                    None if git_commit is None else str(git_commit),
                    json.dumps(self.runtime_configuration, sort_keys=True),
                    None if notes is None else str(notes),
                ),
            )
            for subject in normalized_subjects:
                connection.execute(
                    """
                    INSERT INTO operator_test_subjects (
                        run_id, subject_id, display_name, expected_customer_id
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        subject["subjectId"],
                        subject["displayName"],
                        subject["expectedCustomerId"],
                    ),
                )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM operator_test_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            result = self._run_payload(connection, row)
        self._write_manifest(result)
        return result

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM operator_test_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return None if row is None else self._run_payload(connection, row)

    def stop_run(self, run_id: str) -> dict[str, Any]:
        self.flush_events()
        now_ms = time.time_ns() // 1_000_000
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM operator_test_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise LookupError(f"Unknown test run: {run_id}")
            if row["status"] != "active":
                connection.rollback()
                raise RuntimeError(f"Test run {run_id} is already stopped.")
            connection.execute(
                """
                UPDATE operator_test_runs
                SET status='stopped', stopped_at_unix_ms=?
                WHERE run_id=?
                """,
                (now_ms, run_id),
            )
            connection.execute(
                """
                UPDATE operator_physical_visits
                SET left_at_unix_ms=COALESCE(left_at_unix_ms, ?)
                WHERE run_id=? AND left_at_unix_ms IS NULL
                """,
                (now_ms, run_id),
            )
            connection.commit()
            result_row = connection.execute(
                "SELECT * FROM operator_test_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            result = self._run_payload(connection, result_row)
        self.export_run(run_id)
        return result

    def append_event(self, event: OperatorEvent) -> None:
        with self._connection() as connection:
            active = connection.execute(
                "SELECT run_id FROM operator_test_runs WHERE status='active'"
            ).fetchone()
            if active is None:
                return
            connection.execute(
                """
                INSERT OR IGNORE INTO operator_system_events (
                    event_id, run_id, event_type, occurred_at_unix_ms,
                    host_synced_seconds, camera_index, device_id, track_id,
                    visit_id, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    active["run_id"],
                    event.event_type,
                    event.occurred_at_unix_milliseconds,
                    event.host_synced_seconds,
                    event.camera_index,
                    event.device_id,
                    event.track_id,
                    event.visit_id,
                    json.dumps(event.as_payload(), sort_keys=True),
                ),
            )

    def enqueue_event(self, event: OperatorEvent) -> None:
        if self._writer_closed:
            return
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            print(
                "OPERATOR_EVENT_QUEUE_FULL "
                f"event_id={event.event_id} event_type={event.event_type}"
            )

    def flush_events(self) -> None:
        self._event_queue.join()
        if self._writer_error is not None:
            raise RuntimeError("Operator event writer failed.") from self._writer_error

    def close(self) -> None:
        if self._writer_closed:
            return
        self.flush_events()
        self._writer_closed = True
        self._event_queue.put(None)
        self._writer_thread.join(timeout=5.0)
        if self._writer_thread.is_alive():
            print("Warning: operator event writer did not stop before timeout.")

    def _event_writer(self) -> None:
        while True:
            event = self._event_queue.get()
            try:
                if event is None:
                    return
                self.append_event(event)
            except BaseException as exc:
                self._writer_error = exc
                print(
                    "OPERATOR_EVENT_PERSIST_ERROR "
                    f"event_id={getattr(event, 'event_id', 'none')} error={exc}"
                )
            finally:
                self._event_queue.task_done()

    def create_annotation(
        self,
        run_id: str,
        payload: Mapping[str, Any],
        *,
        system_snapshot: Mapping[str, Any],
        observation_reference: ObservationReference | None,
    ) -> dict[str, Any]:
        annotation_type = str(payload.get("annotationType", "")).strip()
        if annotation_type not in ANNOTATION_TYPES:
            raise ValueError(f"Unsupported annotationType: {annotation_type}")
        if annotation_type in {
            "system_event_correct",
            "system_event_incorrect",
        }:
            system_event_id = payload.get("systemEventId")
            if (
                isinstance(system_event_id, bool)
                or not isinstance(system_event_id, int)
                or system_event_id <= 0
            ):
                raise ValueError(
                    f"{annotation_type} requires a positive systemEventId."
                )
        if annotation_type in {
            "world_state_claim_correct",
            "world_state_claim_incorrect",
        }:
            state_ref = payload.get("worldStateRef")
            if not isinstance(state_ref, Mapping) or not state_ref.get("snapshotId"):
                raise ValueError(f"{annotation_type} requires worldStateRef.snapshotId.")
            if not isinstance(payload.get("claim"), str) or not str(payload["claim"]).strip():
                raise ValueError(f"{annotation_type} requires claim.")
        if annotation_type == "subject_visit_mapping":
            mapped_visit_id = payload.get("visitId")
            if (
                isinstance(mapped_visit_id, bool)
                or not isinstance(mapped_visit_id, int)
                or mapped_visit_id <= 0
            ):
                raise ValueError("subject_visit_mapping requires a positive visitId.")
        if (
            annotation_type in OBSERVATION_ANNOTATION_TYPES
            and observation_reference is None
        ):
            raise ValueError(
                f"{annotation_type} requires an observationRef."
            )
        subject_id_value = payload.get("subjectId")
        subject_id = (
            None
            if subject_id_value in (None, "")
            else self._slug(str(subject_id_value))
        )
        if annotation_type != "note" and subject_id is None:
            raise ValueError("subjectId is required.")
        now_ms = time.time_ns() // 1_000_000
        client_time = payload.get("clientRecordedAtUnixMilliseconds")
        physical_visit_id: str | None = None

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT status FROM operator_test_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None:
                connection.rollback()
                raise LookupError(f"Unknown test run: {run_id}")
            if run["status"] != "active":
                connection.rollback()
                raise RuntimeError(f"Test run {run_id} is not active.")
            if subject_id is not None:
                subject = connection.execute(
                    """
                    SELECT 1 FROM operator_test_subjects
                    WHERE run_id=? AND subject_id=?
                    """,
                    (run_id, subject_id),
                ).fetchone()
                if subject is None:
                    connection.rollback()
                    raise LookupError(f"Unknown test subject: {subject_id}")

                open_visit = connection.execute(
                    """
                    SELECT *
                    FROM operator_physical_visits
                    WHERE run_id=? AND subject_id=? AND left_at_unix_ms IS NULL
                    ORDER BY ordinal DESC
                    LIMIT 1
                    """,
                    (run_id, subject_id),
                ).fetchone()
                if annotation_type == "physical_entry":
                    if open_visit is not None:
                        connection.rollback()
                        raise RuntimeError(
                            f"Subject {subject_id} already has an open physical visit."
                        )
                    ordinal_row = connection.execute(
                        """
                        SELECT COALESCE(MAX(ordinal), 0) + 1 AS value
                        FROM operator_physical_visits
                        WHERE run_id=? AND subject_id=?
                        """,
                        (run_id, subject_id),
                    ).fetchone()
                    ordinal = int(ordinal_row["value"])
                    physical_visit_id = f"{run_id}:{subject_id}:visit-{ordinal}"
                    connection.execute(
                        """
                        INSERT INTO operator_physical_visits (
                            physical_visit_id, run_id, subject_id, ordinal,
                            entered_at_unix_ms, mapping_status
                        )
                        VALUES (?, ?, ?, ?, ?, 'unmapped')
                        """,
                        (
                            physical_visit_id,
                            run_id,
                            subject_id,
                            ordinal,
                            now_ms,
                        ),
                    )
                elif annotation_type == "physical_leave":
                    if open_visit is None:
                        connection.rollback()
                        raise RuntimeError(
                            f"Subject {subject_id} has no open physical visit."
                        )
                    physical_visit_id = str(open_visit["physical_visit_id"])
                    connection.execute(
                        """
                        UPDATE operator_physical_visits
                        SET left_at_unix_ms=?
                        WHERE physical_visit_id=?
                        """,
                        (now_ms, physical_visit_id),
                    )
                elif open_visit is not None:
                    physical_visit_id = str(open_visit["physical_visit_id"])

                if annotation_type == "subject_visit_mapping":
                    if open_visit is not None:
                        physical_visit_id = str(open_visit["physical_visit_id"])
                        connection.execute(
                            """
                            UPDATE operator_physical_visits
                            SET mapped_system_visit_id=?, mapping_status='confirmed'
                            WHERE physical_visit_id=?
                            """,
                            (int(payload["visitId"]), physical_visit_id),
                        )

            cursor = connection.execute(
                """
                INSERT INTO operator_annotations (
                    run_id, annotation_type, subject_id, physical_visit_id,
                    received_at_unix_ms, client_recorded_at_unix_ms,
                    observation_reference_json, system_snapshot_json,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    annotation_type,
                    subject_id,
                    physical_visit_id,
                    now_ms,
                    None if client_time is None else int(client_time),
                    (
                        None
                        if observation_reference is None
                        else json.dumps(
                            observation_reference.payload(), sort_keys=True
                        )
                    ),
                    json.dumps(system_snapshot, sort_keys=True),
                    json.dumps(dict(payload), sort_keys=True),
                ),
            )
            annotation_id = int(cursor.lastrowid)
            connection.commit()

        return {
            "annotationId": annotation_id,
            "runId": run_id,
            "annotationType": annotation_type,
            "subjectId": subject_id,
            "physicalVisitId": physical_visit_id,
            "receivedAtUnixMilliseconds": now_ms,
            "payload": dict(payload),
            "observationRef": (
                None
                if observation_reference is None
                else observation_reference.payload()
            ),
        }

    def subject_visit_mapping(self, run_id: str, subject_id: str) -> int | None:
        """Return the confirmed/current visit mapping for an operator subject."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT mapped_system_visit_id
                FROM operator_physical_visits
                WHERE run_id=? AND subject_id=? AND left_at_unix_ms IS NULL
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                (run_id, subject_id),
            ).fetchone()
            if row is not None and row["mapped_system_visit_id"] is not None:
                return int(row["mapped_system_visit_id"])
            annotation = connection.execute(
                """
                SELECT observation_reference_json, payload_json
                FROM operator_annotations
                WHERE run_id=? AND subject_id=?
                  AND annotation_type IN (
                    'subject_visit_mapping', 'observation_is_subject'
                  )
                ORDER BY annotation_id DESC
                LIMIT 1
                """,
                (run_id, subject_id),
            ).fetchone()
        if annotation is None:
            return None
        payload = json.loads(str(annotation["payload_json"]))
        if payload.get("visitId") is not None:
            return int(payload["visitId"])
        reference_json = annotation["observation_reference_json"]
        if reference_json:
            reference = json.loads(str(reference_json))
            if reference.get("observedVisitId") is not None:
                return int(reference["observedVisitId"])
        return None

    def update_annotation_evidence(
        self,
        annotation_id: int,
        *,
        evidence_path: Path,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE operator_annotations
                SET evidence_path=?
                WHERE annotation_id=?
                """,
                (str(evidence_path), annotation_id),
            )

    def run_rows(
        self,
        run_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._connection() as connection:
            annotations = [
                self._annotation_payload(row)
                for row in connection.execute(
                    """
                    SELECT * FROM operator_annotations
                    WHERE run_id=?
                    ORDER BY received_at_unix_ms, annotation_id
                    """,
                    (run_id,),
                ).fetchall()
            ]
            events = [
                json.loads(str(row["payload_json"]))
                for row in connection.execute(
                    """
                    SELECT payload_json FROM operator_system_events
                    WHERE run_id=?
                    ORDER BY occurred_at_unix_ms, event_id
                    """,
                    (run_id,),
                ).fetchall()
            ]
        return annotations, events

    def physical_visits(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM operator_physical_visits
                WHERE run_id=?
                ORDER BY subject_id, ordinal
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "physicalVisitId": row["physical_visit_id"],
                "runId": row["run_id"],
                "subjectId": row["subject_id"],
                "ordinal": int(row["ordinal"]),
                "enteredAtUnixMilliseconds": int(row["entered_at_unix_ms"]),
                "leftAtUnixMilliseconds": row["left_at_unix_ms"],
                "mappedSystemVisitId": row["mapped_system_visit_id"],
                "mappingStatus": row["mapping_status"],
            }
            for row in rows
        ]

    def subjects(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM operator_test_subjects
                WHERE run_id=?
                ORDER BY subject_id
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "subjectId": row["subject_id"],
                "displayName": row["display_name"],
                "expectedCustomerId": row["expected_customer_id"],
            }
            for row in rows
        ]

    def save_analysis(
        self,
        run_id: str,
        results: Sequence[AnalysisResult],
        *,
        physical_visit_mappings: Mapping[str, int | None],
    ) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM operator_analysis_results WHERE run_id=?", (run_id,)
            )
            for result in results:
                connection.execute(
                    """
                    INSERT INTO operator_analysis_results (
                        run_id, rule_id, status, expected_annotation_id,
                        matched_event_id, latency_ms, summary, details_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result.rule_id,
                        result.status,
                        result.expected_annotation_id,
                        result.matched_event_id,
                        result.latency_milliseconds,
                        result.summary,
                        json.dumps(dict(result.details), sort_keys=True),
                    ),
                )
            for physical_visit_id, system_visit_id in physical_visit_mappings.items():
                connection.execute(
                    """
                    UPDATE operator_physical_visits
                    SET mapped_system_visit_id=?,
                        mapping_status=?
                    WHERE physical_visit_id=?
                    """,
                    (
                        system_visit_id,
                        "mapped" if system_visit_id is not None else "unmapped",
                        physical_visit_id,
                    ),
                )
            connection.commit()

    def analysis_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM operator_analysis_results
                WHERE run_id=?
                ORDER BY result_id
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "resultId": int(row["result_id"]),
                "ruleId": row["rule_id"],
                "status": row["status"],
                "expectedAnnotationId": row["expected_annotation_id"],
                "matchedEventId": row["matched_event_id"],
                "latencyMilliseconds": row["latency_ms"],
                "summary": row["summary"],
                "details": json.loads(str(row["details_json"])),
            }
            for row in rows
        ]

    def load_visit_states(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='visits'
                """
            ).fetchone()
            if table is None:
                return []
            rows = connection.execute(
                """
                SELECT visit_id, status, origin, shopping_customer_id,
                       last_seen_host_seconds, last_device_id, last_track_id
                FROM visits
                ORDER BY visit_id
                """
            ).fetchall()
        return [
            {
                "visitId": int(row["visit_id"]),
                "status": row["status"],
                "origin": row["origin"],
                "customerId": row["shopping_customer_id"],
                "lastSeenHostSeconds": row["last_seen_host_seconds"],
                "lastDeviceId": row["last_device_id"],
                "lastTrackId": row["last_track_id"],
            }
            for row in rows
        ]

    def export_run(
        self,
        run_id: str,
        *,
        report: Mapping[str, Any] | None = None,
    ) -> Path:
        run = self.get_run(run_id)
        if run is None:
            raise LookupError(f"Unknown test run: {run_id}")
        annotations, events = self.run_rows(run_id)
        directory = self.runs_root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "manifest.json", run)
        self._write_json(directory / "subjects.json", self.subjects(run_id))
        self._write_json(
            directory / "physical-visits.json", self.physical_visits(run_id)
        )
        self._write_jsonl(directory / "annotations.jsonl", annotations)
        self._write_jsonl(directory / "system-events.jsonl", events)
        if report is not None:
            self._write_json(directory / "report.json", report)
            (directory / "report.md").write_text(
                self._report_markdown(report), encoding="utf-8"
            )
        return directory

    def evidence_directory(self, run_id: str) -> Path:
        directory = self.runs_root / run_id / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_plane_crossing_evidence(
        self,
        *,
        filename_stem: str,
        jpeg: bytes,
        metadata: Mapping[str, Any],
    ) -> str | None:
        run = self.active_run()
        if run is None:
            return None
        run_id = str(run["runId"])
        safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", filename_stem).strip("-")
        directory = self.evidence_directory(run_id)
        image_path = directory / f"{safe_stem}.jpg"
        metadata_path = directory / f"{safe_stem}.json"
        image_path.write_bytes(jpeg)
        self._write_json(metadata_path, metadata)
        return image_path.relative_to(self.runs_root / run_id).as_posix()

    @staticmethod
    def _run_payload(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        run_id = str(row["run_id"])
        subjects = connection.execute(
            """
            SELECT subject_id, display_name, expected_customer_id
            FROM operator_test_subjects
            WHERE run_id=?
            ORDER BY subject_id
            """,
            (run_id,),
        ).fetchall()
        return {
            "runId": run_id,
            "scenario": row["scenario"],
            "verifier": row["verifier"],
            "status": row["status"],
            "startedAtUnixMilliseconds": int(row["started_at_unix_ms"]),
            "stoppedAtUnixMilliseconds": row["stopped_at_unix_ms"],
            "gitCommit": row["git_commit"],
            "configuration": json.loads(str(row["configuration_json"])),
            "notes": row["notes"],
            "subjects": [
                {
                    "subjectId": subject["subject_id"],
                    "displayName": subject["display_name"],
                    "expectedCustomerId": subject["expected_customer_id"],
                }
                for subject in subjects
            ],
        }

    @staticmethod
    def _annotation_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "annotationId": int(row["annotation_id"]),
            "runId": row["run_id"],
            "annotationType": row["annotation_type"],
            "subjectId": row["subject_id"],
            "physicalVisitId": row["physical_visit_id"],
            "receivedAtUnixMilliseconds": int(row["received_at_unix_ms"]),
            "clientRecordedAtUnixMilliseconds": row[
                "client_recorded_at_unix_ms"
            ],
            "observationRef": (
                None
                if row["observation_reference_json"] is None
                else json.loads(str(row["observation_reference_json"]))
            ),
            "systemSnapshot": json.loads(str(row["system_snapshot_json"])),
            "evidencePath": row["evidence_path"],
            "payload": json.loads(str(row["payload_json"])),
        }

    def _write_manifest(self, run: Mapping[str, Any]) -> None:
        directory = self.runs_root / str(run["runId"])
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "manifest.json", run)

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, payloads: Sequence[Mapping[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
            encoding="utf-8",
        )

    @staticmethod
    def _report_markdown(report: Mapping[str, Any]) -> str:
        lines = [
            f"# Test Run {report['runId']}",
            "",
            f"Status: **{report['status']}**",
            "",
            "| Result | Rule | Summary | Latency |",
            "|---|---|---|---:|",
        ]
        for result in report.get("results", []):
            latency = result.get("latencyMilliseconds")
            lines.append(
                "| {status} | `{rule}` | {summary} | {latency} |".format(
                    status=str(result["status"]).upper(),
                    rule=result["ruleId"],
                    summary=str(result["summary"]).replace("|", "\\|"),
                    latency="" if latency is None else f"{latency} ms",
                )
            )
        return "\n".join(lines) + "\n"
