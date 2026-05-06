"""FastAPI application factory and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from python.agent.tools import ToolExecutor
from python.api._limiter import limiter
from python.api.routes import agent, ingest, query, scene
from python.api.schemas import settings
from python.feature_store.persistence import IndexPersistence
from python.utils.errors import IngestionError, QueryError, SceneQueryError
from python.utils.ipc import ViewerBridge
from python.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Module-level singletons (set during lifespan startup)
_persistence: IndexPersistence | None = None
_viewer_bridge: ViewerBridge | None = None
_tool_executor: ToolExecutor | None = None


def get_persistence() -> IndexPersistence:
    if _persistence is None:
        raise RuntimeError("App not started — persistence not initialized")
    return _persistence


def get_viewer_bridge() -> ViewerBridge | None:
    return _viewer_bridge


def get_tool_executor() -> ToolExecutor:
    """Return the shared ToolExecutor singleton (Factor 5: unify state).

    Sharing one instance means the Searcher's in-memory index cache is
    preserved across all requests instead of being discarded after each call.
    """
    if _tool_executor is None:
        raise RuntimeError("App not started — tool executor not initialized")
    return _tool_executor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _persistence, _viewer_bridge, _tool_executor

    configure_logging(level=settings.log_level)
    logger.info("scene-query starting up")

    _persistence = IndexPersistence(store_root=settings.index_root)

    _viewer_bridge = ViewerBridge(socket_path=settings.socket_path)
    await _viewer_bridge.connect()

    _tool_executor = ToolExecutor(persistence=_persistence, viewer_bridge=_viewer_bridge)

    yield  # Application runs here

    logger.info("scene-query shutting down")
    if _viewer_bridge:
        await _viewer_bridge.close()


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    response = JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    response = request.app.state.limiter._inject_headers(
        response, request.state.view_rate_limit
    )
    return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="scene-query",
        description="Natural language queries over 3D scenes",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Rate limiter state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Register routes
    prefix = "/api/v1"
    app.include_router(ingest.router, prefix=prefix, tags=["ingestion"])
    app.include_router(query.router, prefix=prefix, tags=["query"])
    app.include_router(scene.router, prefix=prefix, tags=["scene"])
    app.include_router(agent.router, prefix=prefix, tags=["agent"])

    # Error handlers
    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(request, exc: IngestionError):
        return JSONResponse(status_code=422, content={"error": "ingestion_failed", "detail": str(exc)})

    @app.exception_handler(QueryError)
    async def query_error_handler(request, exc: QueryError):
        return JSONResponse(status_code=500, content={"error": "query_failed", "detail": str(exc)})

    @app.exception_handler(SceneQueryError)
    async def generic_scene_error_handler(request, exc: SceneQueryError):
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "See server logs"})

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("python.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
