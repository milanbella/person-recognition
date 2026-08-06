from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceAgentConfig:
    openai_api_key: str
    operator_api_base_url: str
    operator_api_token: str
    state_db: Path
    model: str = "gpt-realtime-2.1-mini"
    voice: str = "marin"
    transcription_model: str = "gpt-realtime-whisper"
    idle_timeout_seconds: float = 120.0
    max_session_seconds: float = 2700.0
    event_poll_seconds: float = 0.5
    max_queued_events: int = 100
    allowed_origins: tuple[str, ...] = ()
    retain_transcripts: bool = True
    announce_major_events: bool = False

    def __post_init__(self) -> None:
        if not self.openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is required.")
        if not self.operator_api_token.strip():
            raise ValueError("An operator API token is required.")
        if self.retain_transcripts and not self.transcription_model.strip():
            raise ValueError("A transcription model is required when retaining transcripts.")
        if self.idle_timeout_seconds <= 0:
            raise ValueError("Voice idle timeout must be greater than zero.")
        if self.max_session_seconds <= 0:
            raise ValueError("Voice maximum session duration must be greater than zero.")
        if self.event_poll_seconds <= 0:
            raise ValueError("Voice event polling interval must be greater than zero.")
        if self.max_queued_events <= 0:
            raise ValueError("Voice event queue size must be greater than zero.")
