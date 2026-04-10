"""Agent settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Configuration for the Qwen/Ollama agent loop."""

    ollama_base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    max_iterations: int = 10
    temperature: float = 0.0

    model_config = {"env_prefix": "SQ_AGENT_"}


agent_settings = AgentSettings()
