from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from voice_agent.audit_store import VoiceAuditStore
from voice_agent.config import VoiceAgentConfig
from voice_agent.operator_client import OperatorApiClient, OperatorApiError
from voice_agent.realtime_session import (
    OpenAIRealtimeClient,
    RealtimeCall,
    RealtimeVoiceSession,
    realtime_session_config,
)


LOGGER = logging.getLogger("shop_voice_agent")


class VoiceSessionManager:
    def __init__(
        self,
        config: VoiceAgentConfig,
        *,
        openai_client: OpenAIRealtimeClient | None = None,
        operator_client: OperatorApiClient | None = None,
        audit_store: VoiceAuditStore | None = None,
    ) -> None:
        self.config = config
        self.openai_client = openai_client or OpenAIRealtimeClient(config.openai_api_key)
        self.operator_client = operator_client or OperatorApiClient(
            config.operator_api_base_url,
            config.operator_api_token,
        )
        self.audit_store = audit_store or VoiceAuditStore(config.state_db)
        self._sessions: dict[str, RealtimeVoiceSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        offer_sdp: str,
        *,
        operator_id: str,
    ) -> tuple[RealtimeCall, RealtimeVoiceSession]:
        async with self._lock:
            self._sessions = {
                session_id: session
                for session_id, session in self._sessions.items()
                if session.status != "ended"
            }
            if any(session.status != "ended" for session in self._sessions.values()):
                raise HTTPException(
                    status_code=409,
                    detail="Another shop voice session is already active.",
                )
            try:
                state = await asyncio.to_thread(self.operator_client.state)
            except OperatorApiError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            active_run = state.get("activeRun")
            if not isinstance(active_run, dict) or not active_run.get("runId"):
                raise HTTPException(
                    status_code=409,
                    detail="Start an operator test run before starting voice mode.",
                )
            run_id = str(active_run["runId"])
            voice_session_id = str(uuid.uuid4())
            await asyncio.to_thread(
                self.audit_store.start_session,
                voice_session_id,
                run_id=run_id,
                operator_id=operator_id,
            )
            try:
                call = await asyncio.to_thread(
                    self.openai_client.create_call,
                    offer_sdp,
                    realtime_session_config(self.config),
                )
            except Exception:
                await asyncio.to_thread(
                    self.audit_store.end_session,
                    voice_session_id,
                    "openai_call_creation_failed",
                )
                raise
            session = RealtimeVoiceSession(
                voice_session_id=voice_session_id,
                call_id=call.call_id,
                run_id=run_id,
                config=self.config,
                operator_client=self.operator_client,
                audit_store=self.audit_store,
            )
            self._sessions[voice_session_id] = session
            session.start()
            LOGGER.info(
                "VOICE_SESSION_STARTED voice_session_id=%s run_id=%s operator_id=%s",
                voice_session_id,
                run_id,
                operator_id,
            )
            return call, session

    def get(self, voice_session_id: str) -> RealtimeVoiceSession | None:
        return self._sessions.get(voice_session_id)

    async def stop_session(self, voice_session_id: str) -> RealtimeVoiceSession:
        session = self.get(voice_session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown voice session.")
        await session.stop()
        return session

    async def stop_all(self) -> None:
        sessions = [session for session in self._sessions.values() if session.status != "ended"]
        await asyncio.gather(
            *(session.stop("service_shutdown") for session in sessions),
            return_exceptions=True,
        )


def create_voice_app(
    config: VoiceAgentConfig,
    *,
    manager: VoiceSessionManager | None = None,
) -> FastAPI:
    session_manager = manager or VoiceSessionManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await session_manager.stop_all()

    app = FastAPI(
        title="Shop Realtime Voice Agent",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    if config.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Operator-Id"],
            expose_headers=["X-Voice-Session-Id", "X-Voice-Run-Id"],
        )
    app.state.voice_session_manager = session_manager

    def require_auth(authorization: str | None) -> None:
        if authorization != f"Bearer {config.operator_api_token}":
            raise HTTPException(
                status_code=401,
                detail="A valid operator bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/operator/voice/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/operator/voice/sessions")
    async def create_session(
        request: Request,
        authorization: str | None = Header(default=None),
        operator_id: str | None = Header(default=None, alias="X-Operator-Id"),
    ) -> Response:
        require_auth(authorization)
        content_type = request.headers.get("content-type", "")
        if "application/sdp" not in content_type:
            raise HTTPException(status_code=415, detail="Expected application/sdp.")
        offer = (await request.body()).decode("utf-8", errors="strict")
        if not offer.strip() or len(offer) > 256_000:
            raise HTTPException(status_code=422, detail="Invalid SDP offer.")
        try:
            call, session = await session_manager.create_session(
                offer,
                operator_id=(operator_id or "mobile-operator")[:120],
            )
        except RuntimeError as exc:
            LOGGER.exception("VOICE_SESSION_CREATE_FAILED")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(
            call.answer_sdp,
            media_type="application/sdp",
            headers={
                "Cache-Control": "no-store",
                "X-Voice-Session-Id": session.voice_session_id,
                "X-Voice-Run-Id": session.run_id,
            },
        )

    @app.get("/operator/voice/sessions/{voice_session_id}")
    async def get_session(
        voice_session_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_auth(authorization)
        session = session_manager.get(voice_session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown voice session.")
        return session.payload()

    @app.delete("/operator/voice/sessions/{voice_session_id}")
    async def delete_session(
        voice_session_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_auth(authorization)
        session = await session_manager.stop_session(voice_session_id)
        return session.payload()

    return app
