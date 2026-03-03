"""Unit tests for CLIPTextEncoder."""

from __future__ import annotations

import numpy as np
import pytest


def test_encode_returns_correct_shape():
    pytest.importorskip("open_clip")
    from python.query_engine.encoder import CLIPTextEncoder

    encoder = CLIPTextEncoder()
    embedding = encoder.encode("red chair")

    assert embedding.ndim == 1
    assert embedding.shape[0] > 0
    assert embedding.dtype == np.float32


def test_encode_is_normalized():
    pytest.importorskip("open_clip")
    from python.query_engine.encoder import CLIPTextEncoder

    encoder = CLIPTextEncoder()
    embedding = encoder.encode("a wooden table")
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5, f"Embedding not unit norm: {norm}"


def test_batch_encode():
    pytest.importorskip("open_clip")
    from python.query_engine.encoder import CLIPTextEncoder

    encoder = CLIPTextEncoder()
    queries = ["red chair", "blue sofa", "wooden table"]
    embeddings = encoder.encode_batch(queries)

    assert embeddings.shape == (3, embeddings.shape[1])
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_similar_queries_have_high_cosine():
    pytest.importorskip("open_clip")
    from python.query_engine.encoder import CLIPTextEncoder

    encoder = CLIPTextEncoder()
    e1 = encoder.encode("chair")
    e2 = encoder.encode("seat")
    e3 = encoder.encode("automobile")

    sim_related = float(np.dot(e1, e2))
    sim_unrelated = float(np.dot(e1, e3))
    assert sim_related > sim_unrelated, (
        f"'chair'↔'seat' ({sim_related:.3f}) should score higher than 'chair'↔'automobile' ({sim_unrelated:.3f})"
    )
