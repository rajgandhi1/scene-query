"""Per-conversation state: message history and active scene context."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Keep the most recent N messages to stay within the model's context window.
# Each agent step adds ~3-5 messages; 40 gives ~8-13 turns of headroom.
_CONTEXT_WINDOW_MESSAGES = 40


@dataclass
class AgentSession:
    """Holds message history for one conversation with the agent."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.last_used = time.time()

    def add_assistant(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def trim(self, max_messages: int = _CONTEXT_WINDOW_MESSAGES) -> None:
        """Drop oldest messages when history exceeds the context window cap.

        Prevents the context from silently overflowing the model's token limit.
        Called by the agent loop before each LLM request (Factor 3).
        """
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def clear(self) -> None:
        self.messages.clear()
