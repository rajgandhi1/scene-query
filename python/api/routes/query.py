"""POST /api/v1/query — text query endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from python.api.schemas import QueryMatch, QueryRequest, QueryResponse
from python.feature_store.persistence import IndexPersistence
from python.query_engine.encoder import CLIPTextEncoder
from python.query_engine.reranker import SpatialReranker
from python.query_engine.searcher import Searcher
from python.utils.errors import QueryError, SceneNotFoundError
from python.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_scene(request: QueryRequest) -> QueryResponse:
    """
    Query a scene with a text prompt.

    Returns top-k matching primitive IDs with similarity scores and
    optionally sends highlights to the connected viewer.
    """
    t0 = time.perf_counter()
    logger.info("Query scene=%s query='%s' top_k=%d", request.scene_id, request.query, request.top_k)

    try:
        from python.api.app import get_persistence, get_viewer_bridge

        persistence: IndexPersistence = get_persistence()
        searcher = Searcher(persistence)

        # 1. Encode text query
        encoder = CLIPTextEncoder()
        embedding = encoder.encode(request.query)

        # 2. Search
        results = searcher.search(
            scene_id=request.scene_id,
            query_embedding=embedding,
            top_k=request.top_k,
            threshold=request.threshold,
        )

        # 3. Optional spatial reranking
        if request.rerank and results:
            reranker = SpatialReranker()
            # Positions would come from scene registry — placeholder for Phase 2
            import numpy as np
            dummy_positions = np.zeros((max(r.primitive_id for r in results) + 1, 3), dtype=np.float32)
            results = reranker.rerank(results, dummy_positions)

        # 4. Build response
        matches = [
            QueryMatch(
                primitive_id=r.primitive_id,
                score=r.score,
                position_3d=(0.0, 0.0, 0.0),  # populated from scene in Phase 2
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
            scene_id=request.scene_id,
            query=request.query,
            matches=matches,
            query_time_ms=round(elapsed_ms, 2),
            viewer_updated=viewer_updated,
        )

    except SceneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
