"""Unit tests for Pydantic request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_query_request_valid():
    from python.api.schemas import QueryRequest

    req = QueryRequest(scene_id="abc", query="red chair", top_k=50, threshold=0.3)
    assert req.query == "red chair"
    assert req.top_k == 50


def test_query_request_empty_query_fails():
    from python.api.schemas import QueryRequest

    with pytest.raises(ValidationError):
        QueryRequest(scene_id="abc", query="")


def test_query_request_top_k_bounds():
    from python.api.schemas import QueryRequest

    with pytest.raises(ValidationError):
        QueryRequest(scene_id="abc", query="chair", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(scene_id="abc", query="chair", top_k=99999)


def test_query_request_threshold_bounds():
    from python.api.schemas import QueryRequest

    with pytest.raises(ValidationError):
        QueryRequest(scene_id="abc", query="chair", threshold=-0.1)

    with pytest.raises(ValidationError):
        QueryRequest(scene_id="abc", query="chair", threshold=1.1)


def test_ingest_response_serialization():
    from python.api.schemas import IngestResponse

    resp = IngestResponse(
        scene_id="abc123",
        primitive_count=1000,
        feature_dim=512,
        scene_type="point_cloud",
        status="ok",
    )
    data = resp.model_dump()
    assert data["scene_id"] == "abc123"
    assert data["status"] == "ok"
