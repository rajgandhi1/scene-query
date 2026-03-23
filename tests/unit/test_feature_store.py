"""Unit tests for FeatureIndex and IndexPersistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from python.feature_store.index import FeatureIndex
from python.feature_store.persistence import IndexPersistence
from python.utils.errors import FeatureStoreError


def normalized_random(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((n, d)).astype(np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    return feat


def test_build_and_search(random_features):
    index = FeatureIndex("test-scene")
    index.build(random_features)

    assert index.is_built
    assert index.n_primitives == len(random_features)
    assert index.feature_dim == 512

    query = random_features[0]  # exact match should return itself
    ids, scores = index.search(query, top_k=5)

    assert len(ids) == 5
    assert ids[0] == 0, "Top result should be the query itself"
    assert scores[0] > 0.99


def test_build_with_positions(random_features):
    rng = np.random.default_rng(42)
    positions = rng.uniform(-5, 5, (len(random_features), 3)).astype(np.float32)

    index = FeatureIndex("positions-scene")
    index.build(random_features, positions)

    assert index.positions is not None
    assert index.positions.shape == (len(random_features), 3)
    np.testing.assert_array_equal(index.positions, positions)


def test_build_positions_shape_mismatch(random_features):
    bad_positions = np.zeros((len(random_features) + 1, 3), dtype=np.float32)
    index = FeatureIndex("bad-positions")
    with pytest.raises(FeatureStoreError, match="positions shape"):
        index.build(random_features, bad_positions)


def test_positions_none_by_default(random_features):
    index = FeatureIndex("no-positions")
    index.build(random_features)
    assert index.positions is None


def test_threshold_filters_results(random_features):
    index = FeatureIndex("threshold-test")
    index.build(random_features)
    query = normalized_random(1, 512, seed=99)[0]
    ids, scores = index.search(query, top_k=200, threshold=0.99)
    assert all(s >= 0.99 for s in scores)


def test_search_unbuilt_raises():
    index = FeatureIndex("empty")
    with pytest.raises(FeatureStoreError):
        index.search(np.zeros(512, dtype=np.float32))


def test_save_and_load(random_features):
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        persistence = IndexPersistence(Path(tmp))

        fi = FeatureIndex("persist-test")
        fi.build(random_features)
        persistence.save(fi)

        loaded = persistence.load("persist-test")
        assert loaded.n_primitives == fi.n_primitives
        assert loaded.feature_dim == fi.feature_dim
        assert loaded.positions is None  # no positions provided

        ids, scores = loaded.search(random_features[0], top_k=1)
        assert ids[0] == 0


def test_save_and_load_with_positions(random_features):
    pytest.importorskip("faiss")

    rng = np.random.default_rng(7)
    positions = rng.uniform(-10, 10, (len(random_features), 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        persistence = IndexPersistence(Path(tmp))

        fi = FeatureIndex("persist-pos-test")
        fi.build(random_features, positions)
        persistence.save(fi)

        loaded = persistence.load("persist-pos-test")
        assert loaded.positions is not None
        assert loaded.positions.shape == positions.shape
        np.testing.assert_array_almost_equal(loaded.positions, positions)


def test_delete(random_features):
    pytest.importorskip("faiss")

    rng = np.random.default_rng(9)
    positions = rng.uniform(-1, 1, (len(random_features), 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        persistence = IndexPersistence(Path(tmp))
        fi = FeatureIndex("del-test")
        fi.build(random_features, positions)
        persistence.save(fi)

        assert persistence.exists("del-test")
        pos_path = Path(tmp) / "del-test" / "positions.npy"
        assert pos_path.exists()

        persistence.delete("del-test")
        assert not persistence.exists("del-test")
        assert not pos_path.exists()
