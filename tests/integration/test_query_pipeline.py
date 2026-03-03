"""Integration test: query pipeline on an in-memory indexed scene."""

from __future__ import annotations

import numpy as np
import pytest

from python.feature_store.index import FeatureIndex
from python.feature_store.persistence import IndexPersistence
from python.query_engine.searcher import Searcher, SearchResult
from python.utils.errors import SceneNotFoundError


@pytest.mark.integration
def test_searcher_finds_top_result(random_features, tmp_path):
    pytest.importorskip("faiss")

    persistence = IndexPersistence(tmp_path)
    fi = FeatureIndex("query-test")
    fi.build(random_features)
    persistence.save(fi)

    searcher = Searcher(persistence)
    query = random_features[10]
    results = searcher.search("query-test", query, top_k=5)

    assert len(results) > 0
    assert results[0].primitive_id == 10
    assert results[0].score > 0.99


@pytest.mark.integration
def test_searcher_raises_on_missing_scene(tmp_path):
    pytest.importorskip("faiss")
    persistence = IndexPersistence(tmp_path)
    searcher = Searcher(persistence)

    with pytest.raises(SceneNotFoundError):
        searcher.search("ghost-scene", np.zeros(512, dtype=np.float32))


@pytest.mark.integration
def test_threshold_filters(random_features, tmp_path):
    pytest.importorskip("faiss")
    persistence = IndexPersistence(tmp_path)
    fi = FeatureIndex("thresh-test")
    fi.build(random_features)
    persistence.save(fi)

    searcher = Searcher(persistence)
    # With threshold=0.999, only near-exact matches (self) should pass
    results = searcher.search("thresh-test", random_features[5], threshold=0.999, top_k=50)
    assert all(r.score >= 0.999 for r in results)
