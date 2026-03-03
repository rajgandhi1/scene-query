"""Singleton model registry — loads each model once and caches it."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from python.utils.errors import ModelLoadError
from python.utils.logging import get_logger

if TYPE_CHECKING:
    import torch.nn as nn

logger = get_logger(__name__)


class ModelRegistry:
    """
    Singleton that holds all heavy ML models in memory.

    Prevents redundant GPU allocations when multiple pipeline stages
    need the same model (e.g., CLIP used in both lifting and query encoding).

    Usage:
        registry = ModelRegistry()
        clip = registry.get("clip")
    """

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._models: dict[str, Any] = {}
                instance._init_lock = threading.Lock()
                cls._instance = instance
        return cls._instance

    def get(self, name: str) -> Any:
        """
        Return a loaded model by name, loading it on first access.

        Args:
            name: Model key (e.g. "clip", "sam", "grounding_dino", "depth_anything").

        Returns:
            The loaded model object.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        if name not in self._models:
            with self._init_lock:
                # Double-checked locking
                if name not in self._models:
                    logger.info("Loading model: %s", name)
                    self._models[name] = self._load(name)
                    logger.info("Model loaded: %s", name)
        return self._models[name]

    def _load(self, name: str) -> Any:
        from python.models.loaders import ModelFactory
        return ModelFactory.create(name)

    def unload(self, name: str) -> None:
        """Remove a model from the registry to free memory."""
        if name in self._models:
            del self._models[name]
            logger.info("Unloaded model: %s", name)

    def loaded_models(self) -> list[str]:
        """Return names of currently loaded models."""
        return list(self._models.keys())

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only)."""
        with cls._lock:
            cls._instance = None
