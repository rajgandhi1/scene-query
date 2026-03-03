"""E2E test: ingest → query → verify highlights (requires full stack)."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_health_ok(api_client):
    """Smoke test: API is up and returns a valid health response."""
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "degraded")


@pytest.mark.e2e
@pytest.mark.skip(reason="Requires scene file on disk and model weights — run manually")
def test_ingest_and_query_point_cloud(api_client, tmp_path):
    """
    Full workflow: ingest a PLY file, query it, verify matches returned.

    To run manually:
        SQ_SCENE_ROOT=/data/scenes uv run pytest tests/e2e/ -m e2e -s
    """
    import json
    from pathlib import Path

    # 1. Ingest
    scene_path = Path("/data/scenes/test_cloud.ply")
    ingest_resp = api_client.post(
        "/api/v1/ingest",
        json={"scene_path": str(scene_path), "scene_type": "point_cloud"},
    )
    assert ingest_resp.status_code == 200
    scene_id = ingest_resp.json()["scene_id"]

    # 2. Query
    query_resp = api_client.post(
        "/api/v1/query",
        json={"scene_id": scene_id, "query": "red object", "top_k": 20},
    )
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert len(data["matches"]) > 0
    assert data["query_time_ms"] < 500

    # 3. Clean up
    del_resp = api_client.delete(f"/api/v1/scene/{scene_id}")
    assert del_resp.status_code == 200
