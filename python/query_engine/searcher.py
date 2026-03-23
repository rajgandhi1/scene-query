"""Similarity search over the feature store."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from python.feature_store.index import FeatureIndex
from python.feature_store.persistence import IndexPersistence
from python.utils.errors import FeatureStoreError, QueryError, SceneNotFoundError
from python.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Single primitive match from a similarity search."""

    primitive_id: int
    score: float
    position_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Searcher:
    """
    Executes similarity searches against scene feature indexes.

    Keeps loaded indexes in memory keyed by scene_id for fast repeated queries.
    """

    def __init__(self, persistence: IndexPersistence) -> None:
        self._persistence = persistence
        self._loaded: dict[str, FeatureIndex] = {}

    def search(
        self,
        scene_id: str,
        query_embedding: np.ndarray,
        top_k: int = 100,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar primitives for a query embedding.

        Args:
            scene_id: Target scene.
            query_embedding: (D,) float32 L2-normalized query vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity to include.

        Returns:
            List of SearchResult sorted by score descending.

        Raises:
            SceneNotFoundError: If the scene has no stored index.
            QueryError: If the search operation fails.
        """
        index = self._get_index(scene_id)
        try:
            ids, scores = index.search(query_embedding, top_k, threshold)
            positions = index.positions
            results = []
            for pid, score in zip(ids, scores):
                if positions is not None:
                    p = positions[int(pid)]
                    pos = (float(p[0]), float(p[1]), float(p[2]))
                else:
                    pos = (0.0, 0.0, 0.0)
                results.append(SearchResult(primitive_id=int(pid), score=float(score), position_3d=pos))
            return results
        except Exception as exc:
            raise QueryError(f"Search failed for scene '{scene_id}': {exc}") from exc

    def _get_index(self, scene_id: str) -> FeatureIndex:
        if scene_id not in self._loaded:
            if not self._persistence.exists(scene_id):
                raise SceneNotFoundError(f"No feature index for scene '{scene_id}'")
            self._loaded[scene_id] = self._persistence.load(scene_id)
        return self._loaded[scene_id]

    def get_positions(self, scene_id: str) -> np.ndarray | None:
        """
        Return the (N, 3) positions array for a scene, or None if unavailable.

        The index is loaded from persistence on first access if not already cached.
        """
        return self._get_index(scene_id).positions

    def register_index(self, index: FeatureIndex) -> None:
        """Register an in-memory index (used immediately after ingestion)."""
        self._loaded[index.scene_id] = index

    def evict(self, scene_id: str) -> None:
        """Remove a scene's index from memory."""
        self._loaded.pop(scene_id, None)
