"""Pipeline event bus for progress reporting and monitoring."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class PipelineEventBus:
    """
    Simple synchronous event bus for pipeline stage notifications.

    Decouples pipeline stages from logging, UI progress, and monitoring
    without requiring direct imports between modules.

    Usage:
        event_bus = PipelineEventBus()
        event_bus.subscribe("feature_lifting.progress", my_callback)
        event_bus.emit("feature_lifting.progress", {"step": 5, "total": 100})
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback for a named event."""
        self._listeners[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously registered callback."""
        self._listeners[event] = [
            cb for cb in self._listeners[event] if cb is not callback
        ]

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        """Fire all callbacks registered for the given event."""
        for cb in self._listeners[event]:
            cb(payload)

    def clear(self, event: str | None = None) -> None:
        """Remove all listeners, optionally scoped to one event."""
        if event is None:
            self._listeners.clear()
        else:
            self._listeners.pop(event, None)


# Module-level singleton — import and use directly
event_bus = PipelineEventBus()
