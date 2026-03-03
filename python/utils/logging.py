"""Structured logging configuration for scene-query."""

import logging
import sys
from typing import Any


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure root logger for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, emit JSON lines (suitable for log aggregators).
    """
    fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
        if json_logs
        else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use __name__ as the name in each module."""
    return logging.getLogger(name)


def log_pipeline_event(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    """Convenience callback for subscribing a logger to the event bus."""
    logger.debug("event=%s payload=%s", event, payload)
