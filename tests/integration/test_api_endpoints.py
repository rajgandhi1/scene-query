"""Integration tests for FastAPI endpoints using TestClient."""

from __future__ import annotations

import pytest


def test_health_endpoint(api_client):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "models_loaded" in data
    assert "viewer_connected" in data


def test_query_unknown_scene(api_client):
    response = api_client.post(
        "/api/v1/query",
        json={"scene_id": "nonexistent-scene", "query": "red chair"},
    )
    assert response.status_code == 404


def test_get_unknown_scene(api_client):
    response = api_client.get("/api/v1/scene/nonexistent")
    assert response.status_code == 404


def test_delete_unknown_scene(api_client):
    response = api_client.delete("/api/v1/scene/nonexistent")
    assert response.status_code == 404


def test_query_validates_empty_string(api_client):
    response = api_client.post(
        "/api/v1/query",
        json={"scene_id": "abc", "query": ""},
    )
    assert response.status_code == 422
