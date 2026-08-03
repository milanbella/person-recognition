from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from voice_agent.api import create_voice_app
from voice_agent.config import VoiceAgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Companion OpenAI Realtime voice service for shop walk tests."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--operator-api-base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--operator-api-token")
    parser.add_argument("--operator-api-token-file", type=Path)
    parser.add_argument("--openai-api-key-file", type=Path)
    parser.add_argument("--state-db", type=Path, default=Path("state/shop_state.sqlite"))
    parser.add_argument("--realtime-model", default="gpt-realtime-2.1-mini")
    parser.add_argument("--realtime-voice", default="marin")
    parser.add_argument("--transcription-model", default="gpt-live-transcribe")
    parser.add_argument("--idle-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-session-seconds", type=float, default=2700.0)
    parser.add_argument("--event-poll-seconds", type=float, default=0.5)
    parser.add_argument("--allowed-origin", action="append", default=[])
    parser.add_argument(
        "--disable-transcript-retention",
        action="store_true",
        help="Do not store short operator/assistant transcripts in SQLite.",
    )
    return parser.parse_args()


def _operator_token(args: argparse.Namespace) -> str:
    if args.operator_api_token_file is not None:
        return args.operator_api_token_file.read_text(encoding="utf-8").strip()
    return (args.operator_api_token or os.environ.get("SHOP_OPERATOR_API_TOKEN", "")).strip()


def _openai_api_key(args: argparse.Namespace) -> str:
    if args.openai_api_key_file is not None:
        return args.openai_api_key_file.read_text(encoding="utf-8").strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = VoiceAgentConfig(
        openai_api_key=_openai_api_key(args),
        operator_api_base_url=args.operator_api_base_url,
        operator_api_token=_operator_token(args),
        state_db=args.state_db,
        model=args.realtime_model,
        voice=args.realtime_voice,
        transcription_model=args.transcription_model,
        idle_timeout_seconds=args.idle_timeout_seconds,
        max_session_seconds=args.max_session_seconds,
        event_poll_seconds=args.event_poll_seconds,
        allowed_origins=tuple(args.allowed_origin),
        retain_transcripts=not args.disable_transcript_retention,
    )
    uvicorn.run(create_voice_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
