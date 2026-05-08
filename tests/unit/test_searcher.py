"""Unit tests for Searcher — position lookup and fallback behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python.feature_store.index import FeatureIndex
from python.query_engine.searcher import SearchResult, Searcher
from python.utils.errors import QueryError


def _make_mock_index(n: int = 10, with_positions: bool = True, feature_dim: int = 512) -> MagicMock:
    """Return a mock FeatureIndex that returns predictable search results."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(-5, 5, (n, 3)).astype(np.float32) if with_positions else None

    index = MagicMock()
    index.positions = positions
    index.feature_dim = feature_dim
    # search() returns (ids, scores) numpy arrays
    index.search.return_value = (
        np.array([3, 7, 1], dtype=np.int64),
        np.array([0.95, 0.80, 0.70], dtype=np.float32),
    )
    return index


def test_search_returns_position_3d_from_index():
    """SearchResult.position_3d is populated from the index positions array."""
    mock_index = _make_mock_index(with_positions=True)
    persistence = MagicMock()

    searcher = Searcher(persistence)
    searcher._loaded["scene-1"] = mock_index

    query = np.zeros(512, dtype=np.float32)
    results = searcher.search("scene-1", query, top_k=3)

    assert len(results) == 3
    for result in results:
        pid = result.primitive_id
        expected = tuple(float(v) for v in mock_index.positions[pid])
        assert result.position_3d == pytest.approx(expected)


def test_search_falls_back_to_zeros_when_no_positions():
    """When the index has no positions, position_3d is (0, 0, 0) for all results."""
    mock_index = _make_mock_index(with_positions=False)
    persistence = MagicMock()

    searcher = Searcher(persistence)
    searcher._loaded["scene-2"] = mock_index

    query = np.zeros(512, dtype=np.float32)
    results = searcher.search("scene-2", query, top_k=3)

    for result in results:
        assert result.position_3d == (0.0, 0.0, 0.0)


def test_get_positions_returns_array():
    """get_positions() exposes the index positions array."""
    mock_index = _make_mock_index(with_positions=True)
    persistence = MagicMock()

    searcher = Searcher(persistence)
    searcher._loaded["scene-3"] = mock_index

    positions = searcher.get_positions("scene-3")
    assert positions is not None
    assert positions.shape == (10, 3)


def test_get_positions_returns_none_when_absent():
    """get_positions() returns None for indexes without stored positions."""
    mock_index = _make_mock_index(with_positions=False)
    persistence = MagicMock()

    searcher = Searcher(persistence)
    searcher._loaded["scene-4"] = mock_index

    assert searcher.get_positions("scene-4") is None


def test_search_result_default_position():
    """SearchResult defaults position_3d to (0, 0, 0) without explicit argument."""
    result = SearchResult(primitive_id=5, score=0.9)
    assert result.position_3d == (0.0, 0.0, 0.0)


def test_search_raises_query_error_on_dim_mismatch():
    """Mismatched query/index dim raises QueryError, not a numpy or FAISS exception."""
    pytest.importorskip("faiss")

    rng = np.random.default_rng(0)
    features = rng.standard_normal((50, 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    index = FeatureIndex("dim-mismatch-scene")
    index.build(features)

    persistence = MagicMock()
    searcher = Searcher(persistence)
    searcher._loaded["dim-mismatch-scene"] = index

    wrong_dim_query = np.zeros(768, dtype=np.float32)  # ViT-L/14 dim vs ViT-B/32 index
    with pytest.raises(QueryError, match="768.*512|dim.*mismatch|does not match"):
        searcher.search("dim-mismatch-scene", wrong_dim_query)
