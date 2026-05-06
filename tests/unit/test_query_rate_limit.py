"""Unit tests for /api/v1/query rate limiting."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def rate_limited_client(monkeypatch):
    """TestClient with query_rate_limit=1 and a clean limiter storage."""
    import python.api.schemas as schemas_mod
    monkeypatch.setattr(schemas_mod.settings, "query_rate_limit", 1)

    # Reset the shared limiter storage so prior test runs don't bleed in.
    from python.api._limiter import limiter
    limiter._storage.reset()

    from python.api.app import create_app
    from python.models.registry import ModelRegistry

    ModelRegistry.reset()
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    limiter._storage.reset()


def test_query_rate_limit_429(rate_limited_client: TestClient) -> None:
    """Second request within the same minute must return 429."""
    payload = {"scene_id": "does-not-exist", "query": "red chair"}

    # First request — goes through the limiter, hits 404 (unknown scene).
    r1 = rate_limited_client.post("/api/v1/query", json=payload)
    assert r1.status_code != 429, "First request should not be rate-limited"

    # Second request — limiter should reject it.
    r2 = rate_limited_client.post("/api/v1/query", json=payload)
    assert r2.status_code == 429
    assert r2.json() == {"detail": "rate limit exceeded"}
    assert "retry-after" in r2.headers


def test_other_endpoints_not_rate_limited(rate_limited_client: TestClient) -> None:
    """Rate limit on /query must not affect other endpoints."""
    # Exhaust the quota on /query.
    rate_limited_client.post(
        "/api/v1/query", json={"scene_id": "x", "query": "test"}
    )

    # /health should still respond normally.
    r = rate_limited_client.get("/api/v1/health")
    assert r.status_code == 200
