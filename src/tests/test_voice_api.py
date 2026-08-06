import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from starlette.requests import Request

from voice_agent.api import create_voice_app
from voice_agent.config import VoiceAgentConfig
from voice_agent.realtime_session import (
    RealtimeCall,
    _multipart_body,
    realtime_session_config,
)


class FakeSession:
    voice_session_id = "voice-1"
    run_id = "run-1"

    def payload(self):
        return {"voiceSessionId": self.voice_session_id, "runId": self.run_id, "status": "connected"}


class FakeManager:
    def __init__(self) -> None:
        self.session = FakeSession()
        self.offers = []

    async def create_session(self, offer_sdp, *, operator_id):
        self.offers.append((offer_sdp, operator_id))
        return RealtimeCall("answer-sdp", "rtc_1"), self.session

    def get(self, voice_session_id):
        return self.session if voice_session_id == "voice-1" else None

    async def stop_session(self, voice_session_id):
        return self.session

    async def stop_all(self):
        return None


class VoiceApiTests(unittest.TestCase):
    def test_authenticated_sdp_exchange_does_not_expose_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = VoiceAgentConfig(
                openai_api_key="openai-secret",
                operator_api_base_url="http://operator",
                operator_api_token="operator-secret",
                state_db=Path(temporary) / "state.sqlite",
            )
            manager = FakeManager()
            app = create_voice_app(config, manager=manager)
            endpoint = next(
                route.endpoint
                for route in app.routes
                if getattr(route, "path", None) == "/operator/voice/sessions"
                and "POST" in getattr(route, "methods", set())
            )
            request = _sdp_request("offer-sdp")
            with self.assertRaises(HTTPException) as unauthorized:
                asyncio.run(endpoint(request, None, None))
            self.assertEqual(unauthorized.exception.status_code, 401)
            response = asyncio.run(
                endpoint(
                    _sdp_request("offer-sdp"),
                    "Bearer operator-secret",
                    "Milan",
                )
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.body.decode("utf-8"), "answer-sdp")
            self.assertEqual(response.headers["x-voice-session-id"], "voice-1")
            self.assertNotIn("openai-secret", response.body.decode("utf-8"))
            self.assertEqual(manager.offers, [("offer-sdp", "Milan")])

    def test_multipart_contains_sdp_and_session_json(self) -> None:
        body = _multipart_body(
            "boundary",
            offer_sdp="v=0",
            session_config={"type": "realtime", "model": "test"},
        )
        self.assertIn(b'name="sdp"', body)
        self.assertIn(b"v=0", body)
        self.assertIn(b'name="session"', body)
        self.assertIn(b'"model":"test"', body)

    def test_transcription_is_present_only_when_retention_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retained = VoiceAgentConfig(
                openai_api_key="key",
                operator_api_base_url="http://operator",
                operator_api_token="token",
                state_db=Path(temporary) / "state.sqlite",
            )
            private = VoiceAgentConfig(
                openai_api_key="key",
                operator_api_base_url="http://operator",
                operator_api_token="token",
                state_db=Path(temporary) / "state.sqlite",
                retain_transcripts=False,
            )
        self.assertEqual(
            realtime_session_config(retained)["audio"]["input"]["transcription"]["model"],
            "gpt-realtime-whisper",
        )
        self.assertNotIn(
            "transcription",
            realtime_session_config(private)["audio"]["input"],
        )


if __name__ == "__main__":
    unittest.main()


def _sdp_request(body: str) -> Request:
    consumed = False

    async def receive():
        nonlocal consumed
        if consumed:
            return {"type": "http.disconnect"}
        consumed = True
        return {"type": "http.request", "body": body.encode("utf-8"), "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/operator/voice/sessions",
            "headers": [(b"content-type", b"application/sdp")],
        },
        receive,
    )
