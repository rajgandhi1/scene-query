"""FAISS-backed feature index for per-primitive embeddings."""

from __future__ import annotations

import numpy as np

from python.utils.errors import FeatureStoreError
from python.utils.logging import get_logger

logger = get_logger(__name__)


def build_index(features: np.ndarray) -> "faiss.Index":  # type: ignore[name-defined]
    """
    Build a FAISS index appropriate for the dataset size.

    All vectors must be L2-normalized (cosine similarity = inner product).

    Args:
        features: (N, D) float32 L2-normalized feature matrix.

    Returns:
        A populated FAISS index.
    """
    try:
        import faiss
    except ImportError as exc:
        raise FeatureStoreError("faiss not installed. Run: pip install faiss-cpu") from exc

    n, d = features.shape
    features = features.astype(np.float32)

    if n < 100_000:
        logger.info("Building FlatIP index for %d vectors", n)
        index = faiss.IndexFlatIP(d)
        index.add(features)

    elif n < 1_000_000:
        logger.info("Building IVFFlat index (256 clusters) for %d vectors", n)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, 256, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
        index.add(features)

    else:
        logger.info("Building HNSW index for %d vectors", n)
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(features)

    logger.info("Index built: %d vectors, dim=%d", index.ntotal, d)
    return index


class FeatureIndex:
    """
    Manages a FAISS index for one scene.

    Wraps build, search, and persistence into a single object.
    """

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self._index: "faiss.Index | None" = None  # type: ignore[name-defined]
        self._n_primitives: int = 0
        self._feature_dim: int = 0

    def build(self, features: np.ndarray) -> None:
        """
        Build the index from a feature matrix.

        Args:
            features: (N, D) float32 per-primitive embeddings.
        """
        if features.ndim != 2:
            raise FeatureStoreError(f"Expected 2D array, got shape {features.shape}")
        self._n_primitives, self._feature_dim = features.shape
        self._index = build_index(features)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 100,
        threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the top-k primitives most similar to a query embedding.

        Args:
            query: (D,) float32 L2-normalized query embedding.
            top_k: Number of results to return.
            threshold: Minimum cosine similarity to include.

        Returns:
            primitive_ids: (K,) int64 array of matching primitive IDs.
            scores: (K,) float32 cosine similarity scores.
        """
        if self._index is None:
            raise FeatureStoreError(f"Index for scene '{self.scene_id}' not built yet")

        query = query.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query, k)
        scores, ids = scores[0], ids[0]

        if threshold > 0:
            mask = scores >= threshold
            scores, ids = scores[mask], ids[mask]

        return ids.astype(np.int64), scores.astype(np.float32)

    @property
    def n_primitives(self) -> int:
        return self._n_primitives

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def is_built(self) -> bool:
        return self._index is not None
