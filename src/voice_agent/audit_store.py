from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


class VoiceAuditStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
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
                CREATE TABLE IF NOT EXISTS operator_voice_sessions (
                    voice_session_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    operator_id TEXT NOT NULL,
                    openai_call_id TEXT,
                    started_at_unix_ms INTEGER NOT NULL,
                    ended_at_unix_ms INTEGER,
                    status TEXT NOT NULL,
                    disconnect_reason TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS operator_voice_tool_calls (
                    tool_call_id TEXT PRIMARY KEY,
                    voice_session_id TEXT NOT NULL,
                    run_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    result_json TEXT,
                    status TEXT NOT NULL,
                    created_at_unix_ms INTEGER NOT NULL,
                    completed_at_unix_ms INTEGER,
                    annotation_id INTEGER
                );

                CREATE TABLE IF NOT EXISTS operator_voice_turns (
                    voice_session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    occurred_at_unix_ms INTEGER NOT NULL,
                    PRIMARY KEY (voice_session_id, turn_id, speaker)
                );
                """
            )
            connection.commit()

    def start_session(
        self,
        voice_session_id: str,
        *,
        run_id: str | None,
        operator_id: str,
    ) -> None:
        now_ms = time.time_ns() // 1_000_000
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO operator_voice_sessions (
                    voice_session_id, run_id, operator_id,
                    started_at_unix_ms, status
                ) VALUES (?, ?, ?, ?, 'starting')
                """,
                (voice_session_id, run_id, operator_id, now_ms),
            )
            connection.commit()

    def connect_session(self, voice_session_id: str, call_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE operator_voice_sessions
                SET openai_call_id=?, status='connected'
                WHERE voice_session_id=?
                """,
                (call_id, voice_session_id),
            )
            connection.commit()

    def end_session(self, voice_session_id: str, reason: str) -> None:
        now_ms = time.time_ns() // 1_000_000
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE operator_voice_sessions
                SET status='ended', ended_at_unix_ms=?, disconnect_reason=?
                WHERE voice_session_id=? AND status != 'ended'
                """,
                (now_ms, reason, voice_session_id),
            )
            connection.commit()

    def start_tool_call(
        self,
        tool_call_id: str,
        *,
        voice_session_id: str,
        run_id: str | None,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        now_ms = time.time_ns() // 1_000_000
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT status, result_json, annotation_id
                FROM operator_voice_tool_calls WHERE tool_call_id=?
                """,
                (tool_call_id,),
            ).fetchone()
            if existing is not None:
                return {
                    "status": existing["status"],
                    "result": (
                        None
                        if existing["result_json"] is None
                        else json.loads(str(existing["result_json"]))
                    ),
                    "annotationId": existing["annotation_id"],
                }
            connection.execute(
                """
                INSERT INTO operator_voice_tool_calls (
                    tool_call_id, voice_session_id, run_id, tool_name,
                    arguments_json, status, created_at_unix_ms
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    tool_call_id,
                    voice_session_id,
                    run_id,
                    tool_name,
                    json.dumps(dict(arguments), sort_keys=True),
                    now_ms,
                ),
            )
            connection.commit()
        return None

    def finish_tool_call(
        self,
        tool_call_id: str,
        *,
        status: str,
        result: Mapping[str, Any],
        annotation_id: int | None = None,
    ) -> None:
        now_ms = time.time_ns() // 1_000_000
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE operator_voice_tool_calls
                SET status=?, result_json=?, completed_at_unix_ms=?, annotation_id=?
                WHERE tool_call_id=?
                """,
                (
                    status,
                    json.dumps(dict(result), sort_keys=True),
                    now_ms,
                    annotation_id,
                    tool_call_id,
                ),
            )
            connection.commit()

    def add_transcript(
        self,
        voice_session_id: str,
        *,
        turn_id: str,
        speaker: str,
        transcript: str,
    ) -> None:
        if not transcript.strip():
            return
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO operator_voice_turns (
                    voice_session_id, turn_id, speaker, transcript,
                    occurred_at_unix_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    voice_session_id,
                    turn_id,
                    speaker,
                    transcript.strip(),
                    time.time_ns() // 1_000_000,
                ),
            )
            connection.commit()

    def add_usage(
        self,
        voice_session_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE operator_voice_sessions
                SET input_tokens=input_tokens+?, output_tokens=output_tokens+?
                WHERE voice_session_id=?
                """,
                (input_tokens, output_tokens, voice_session_id),
            )
            connection.commit()

