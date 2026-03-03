"""Optional spatial consistency reranking for query results."""

from __future__ import annotations

import numpy as np

from python.query_engine.searcher import SearchResult
from python.utils.logging import get_logger

logger = get_logger(__name__)


class SpatialReranker:
    """
    Reranks search results by spatial clustering coherence.

    The intuition: true object matches cluster in 3D space. Scattered,
    isolated high-score primitives are likely false positives. This
    reranker boosts clusters and suppresses outliers.

    This is an optional post-processing step — the base search results
    are still returned if reranking is disabled or fails.
    """

    def __init__(self, radius: float = 0.5, min_cluster_size: int = 3) -> None:
        """
        Args:
            radius: Neighbourhood radius for cluster detection (scene units).
            min_cluster_size: Minimum primitives in a cluster to boost its score.
        """
        self.radius = radius
        self.min_cluster_size = min_cluster_size

    def rerank(
        self,
        results: list[SearchResult],
        positions: np.ndarray,  # (N_total_primitives, 3) scene XYZ
    ) -> list[SearchResult]:
        """
        Rerank results using spatial clustering.

        Args:
            results: Initial search results sorted by score.
            positions: 3D positions of all primitives in the scene.

        Returns:
            Reranked results (may be fewer if outliers are filtered).
        """
        if len(results) < self.min_cluster_size:
            return results

        ids = np.array([r.primitive_id for r in results])
        scores = np.array([r.score for r in results])
        pts = positions[ids]  # (K, 3)

        # Count neighbours within radius for each result
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (K, K, 3)
        dists = np.linalg.norm(diffs, axis=-1)  # (K, K)
        neighbour_counts = (dists < self.radius).sum(axis=1) - 1  # exclude self

        # Boost score by neighbour density (log scale to avoid dominating)
        density_bonus = np.log1p(neighbour_counts) * 0.05
        adjusted_scores = scores + density_bonus

        # Sort by adjusted score
        order = np.argsort(-adjusted_scores)
        return [
            SearchResult(primitive_id=int(ids[i]), score=float(adjusted_scores[i]))
            for i in order
        ]
