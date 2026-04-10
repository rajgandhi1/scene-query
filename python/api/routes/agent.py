"""POST /api/v1/agent/chat — conversational agent endpoint."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from python.agent.loop import AgentLoop
from python.agent.schemas import agent_settings
from python.agent.session import AgentSession
from python.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# In-memory session store keyed by session_id (Factor 5: single source of truth).
# The ToolExecutor singleton (with its Searcher cache) lives in app.py, not here.
_sessions: dict[str, AgentSession] = {}

_SESSION_TTL_SECONDS = 3600  # evict sessions idle for more than 1 hour


def _evict_stale_sessions() -> None:
    """Remove sessions that haven't been used within the TTL window."""
    cutoff = time.time() - _SESSION_TTL_SECONDS
    stale = [sid for sid, s in _sessions.items() if s.last_used < cutoff]
    for sid in stale:
        del _sessions[sid]
    if stale:
        logger.debug("Evicted %d stale agent sessions", len(stale))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2048)
    session_id: str | None = None  # omit to start a new session


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@router.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the scene-query agent.

    The agent may call tools (query_scene, highlight_primitives, etc.) before
    returning a natural-language reply. Pass the returned session_id in
    subsequent requests to maintain conversation context.
    """
    from python.api.app import get_tool_executor

    _evict_stale_sessions()

    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = AgentSession(session_id=session_id)

    session = _sessions[session_id]
    loop = AgentLoop(executor=get_tool_executor(), settings=agent_settings)

    try:
        reply = await loop.run(session, request.message)
    except Exception as exc:
        logger.exception("Agent loop error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(reply=reply, session_id=session_id)
