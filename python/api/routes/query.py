"""POST /api/v1/query — text query endpoint."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from python.api._limiter import limiter
from python.api.schemas import QueryMatch, QueryRequest, QueryResponse, settings
from python.feature_store.persistence import IndexPersistence
from python.query_engine.encoder import CLIPTextEncoder
from python.query_engine.reranker import SpatialReranker
from python.query_engine.searcher import Searcher
from python.utils.errors import QueryError, SceneNotFoundError
from python.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _query_rate_limit(*args: Any, **kwargs: Any) -> str:
    """Return the current rate-limit string, read from settings at call time."""
    return f"{settings.query_rate_limit}/minute"


@router.post("/query", response_model=QueryResponse)
@limiter.limit(_query_rate_limit)
async def query_scene(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Query a scene with a text prompt.

    Returns top-k matching primitive IDs with similarity scores and
    optionally sends highlights to the connected viewer.
    """
    t0 = time.perf_counter()
    logger.info("Query scene=%s query='%s' top_k=%d", body.scene_id, body.query, body.top_k)

    try:
        from python.api.app import get_persistence, get_viewer_bridge

        persistence: IndexPersistence = get_persistence()
        searcher = Searcher(persistence)

        # 1. Encode text query
        encoder = CLIPTextEncoder()
        embedding = encoder.encode(body.query)

        # 2. Search
        results = searcher.search(
            scene_id=body.scene_id,
            query_embedding=embedding,
            top_k=body.top_k,
            threshold=body.threshold,
        )

        # 3. Optional spatial reranking
        if body.rerank and results:
            reranker = SpatialReranker()
            positions = searcher.get_positions(body.scene_id)
            if positions is not None:
                results = reranker.rerank(results, positions)
            else:
                logger.warning(
                    "Spatial reranking skipped for scene '%s': no positions stored", body.scene_id
                )

        # 4. Build response
        matches = [
            QueryMatch(
                primitive_id=r.primitive_id,
                score=r.score,
                position_3d=r.position_3d,
            )
            for r in results
        ]

        # 5. Send highlights to viewer (non-blocking, best-effort)
        viewer_updated = False
        bridge = get_viewer_bridge()
        if bridge and bridge.connected and matches:
            viewer_updated = await bridge.highlight(
                primitive_ids=[m.primitive_id for m in matches],
                scores=[m.score for m in matches],
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Query complete: %d matches in %.1fms (viewer=%s)",
            len(matches), elapsed_ms, viewer_updated,
        )

        return QueryResponse(
            scene_id=body.scene_id,
            query=body.query,
            matches=matches,
            query_time_ms=round(elapsed_ms, 2),
            viewer_updated=viewer_updated,
        )

    except SceneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
