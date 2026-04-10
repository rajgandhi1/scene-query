"""Agent loop: Qwen via Ollama + tool execution."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from python.agent.schemas import AgentSettings, agent_settings
from python.agent.session import AgentSession
from python.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from python.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a 3D scene analysis assistant. You help users explore and understand \
3D scenes (Gaussian Splats, NeRFs, point clouds) using natural language.

You have tools to:
- Query scenes by text description (returns matching primitive IDs + positions)
- Count objects matching a description
- Highlight regions in the 3D viewer
- Clear highlights
- Measure distances between primitives
- List available scenes

Guidelines:
- After querying, always tell the user what you found (count + representative positions).
- Highlight results in the viewer when it makes sense.
- If a query returns zero matches, suggest rephrasing or lowering the threshold.
- Be concise — avoid unnecessary prose."""

# Module-level client — reused across all requests (Factor 5: unify state)
_client: AsyncOpenAI | None = None


def get_client(settings: AgentSettings) -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # required by the openai client, unused by Ollama
        )
    return _client


def _compact_error(tool_name: str, exc: Exception) -> str:
    """Format a tool error compactly so it wastes minimal context tokens (Factor 9)."""
    return f"[tool_error] {tool_name}: {type(exc).__name__}: {exc}"


class AgentLoop:
    """
    Orchestrates multi-step tool use over a conversation session.

    Uses Qwen (via Ollama's OpenAI-compatible endpoint) for planning.
    Runs tool calls in sequence until the model returns a plain-text reply
    or the iteration limit is reached.
    """

    def __init__(self, executor: ToolExecutor, settings: AgentSettings = agent_settings) -> None:
        self._executor = executor
        self._settings = settings
        self._client = get_client(settings)

    async def run(self, session: AgentSession, user_message: str) -> str:
        """
        Process a user message and return the agent's final text reply.

        Mutates session by appending all messages (user, assistant, tool results).
        Trims context window before each LLM call to prevent token overflow.
        """
        session.add_user(user_message)

        for step in range(self._settings.max_iterations):
            # Factor 3: own the context window — trim before sending to LLM
            session.trim()

            response = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    *session.messages,
                ],
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=self._settings.temperature,
            )

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                # Final text response
                reply = msg.content or ""
                session.add_assistant(reply)
                return reply

            # Serialize tool calls for the session (dicts, not openai objects)
            tool_calls_data: list[dict[str, Any]] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            session.add_assistant(msg.content, tool_calls_data)

            for tc in msg.tool_calls:
                logger.info("Agent step=%d tool=%s", step, tc.function.name)
                try:
                    args: dict[str, Any] = json.loads(tc.function.arguments)
                    result = await self._executor.execute(tc.function.name, args)
                    result_str = json.dumps(result)
                except Exception as exc:
                    # Factor 9: compact errors — don't flood context with tracebacks
                    result_str = _compact_error(tc.function.name, exc)
                    logger.warning("Tool %s raised: %s", tc.function.name, exc)

                session.add_tool_result(tc.id, result_str)

        return "Reached the maximum number of reasoning steps. Please rephrase your request."
