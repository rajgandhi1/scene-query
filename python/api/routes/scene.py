"""GET/DELETE /api/v1/scene — scene metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from python.api.schemas import HealthResponse, SceneMetadata
from python.models.registry import ModelRegistry
from python.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/scene/{scene_id}", response_model=SceneMetadata)
async def get_scene(scene_id: str) -> SceneMetadata:
    """Return metadata for a registered scene."""
    from python.api.routes.ingest import get_scene_registry

    registry = get_scene_registry()
    meta = await registry.get(scene_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
    return SceneMetadata(**meta)


@router.get("/scene/{scene_id}/features")
async def get_scene_features(scene_id: str) -> dict:
    """Return feature store statistics for a scene."""
    from python.api.app import get_persistence
    from python.api.routes.ingest import get_scene_registry

    registry = get_scene_registry()
    meta = await registry.get(scene_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")

    persistence = get_persistence()
    return {
        "scene_id": scene_id,
        "index_exists": persistence.exists(scene_id),
        "primitive_count": meta["primitive_count"],
        "feature_dim": meta["feature_dim"],
    }


@router.delete("/scene/{scene_id}")
async def delete_scene(scene_id: str) -> dict:
    """Remove a scene and its feature index."""
    from python.api.app import get_persistence
    from python.api.routes.ingest import get_scene_registry

    registry = get_scene_registry()
    if not await registry.contains(scene_id):
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")

    persistence = get_persistence()
    persistence.delete(scene_id)
    await registry.delete(scene_id)
    logger.info("Deleted scene: %s", scene_id)
    return {"status": "ok", "scene_id": scene_id}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — lists loaded models and viewer connectivity."""
    from python.api.app import get_viewer_bridge

    model_registry = ModelRegistry()
    bridge = get_viewer_bridge()
    return HealthResponse(
        status="ok",
        models_loaded=model_registry.loaded_models(),
        viewer_connected=bridge.connected if bridge else False,
    )
