"""Integration test: full ingestion on a synthetic scene."""

from __future__ import annotations

import numpy as np
import pytest

from python.feature_store.index import FeatureIndex
from python.ingestion.loaders import PointCloud


@pytest.mark.integration
def test_feature_index_round_trip(random_features):
    """Build an index from features and verify search returns the correct top-1."""
    pytest.importorskip("faiss")

    index = FeatureIndex("integration-test")
    index.build(random_features)

    query = random_features[42]
    ids, scores = index.search(query, top_k=1)
    assert ids[0] == 42, f"Top-1 should be query itself, got {ids[0]}"
    assert scores[0] > 0.99


@pytest.mark.integration
def test_point_cloud_ingestion_produces_correct_feature_shape(tiny_point_cloud):
    """Feature lifting on a point cloud produces (N, D) float32 output."""
    from python.api.routes.ingest import _lift_features
    from python.api.schemas import IngestRequest, IngestionConfig

    class _FakeRequest:
        scene_type = "point_cloud"

    features = _lift_features(tiny_point_cloud, _FakeRequest())
    assert features.shape == (len(tiny_point_cloud), 512)
    assert features.dtype == np.float32

    # Check normalization
    norms = np.linalg.norm(features, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
