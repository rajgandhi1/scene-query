"""POST /api/v1/ingest — scene ingestion endpoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from python.api.schemas import IngestRequest, IngestResponse
from python.feature_lifting.clip_extractor import CLIPExtractor
from python.feature_lifting.feature_projector import ProjectorFactory
from python.feature_store.index import FeatureIndex
from python.feature_store.persistence import IndexPersistence
from python.ingestion.loaders import SceneLoaderFactory
from python.ingestion.validators import SceneValidator
from python.utils.errors import IngestionError, ValidationError
from python.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

_scene_registry: dict[str, dict] = {}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_scene(request: IngestRequest) -> IngestResponse:
    """
    Ingest a scene file: validate → load → extract features → index.

    This is a synchronous endpoint for Phase 1. Phase 3 will move this
    to an async task queue.
    """
    scene_id = str(uuid.uuid4())
    logger.info("Ingesting scene %s: %s", scene_id, request.scene_path)

    try:
        # 1. Validate
        validator = SceneValidator()
        validator.validate(request.scene_path, request.scene_type)

        # 2. Load scene
        scene = SceneLoaderFactory.load(request.scene_path)
        logger.info("Scene loaded: %d primitives", len(scene))

        # 3. Feature lifting
        # Phase 1: image-based lifting requires associated images.
        # For pose-free point clouds, we use color-based pseudo-features.
        features = _lift_features(scene, request)

        # 4. Build and persist index
        from python.api.app import get_persistence
        persistence: IndexPersistence = get_persistence()

        feature_index = FeatureIndex(scene_id)
        feature_index.build(features)
        persistence.save(feature_index)

        # 5. Register scene metadata
        _scene_registry[scene_id] = {
            "scene_id": scene_id,
            "scene_type": request.scene_type,
            "primitive_count": len(scene),
            "feature_dim": features.shape[1],
            "source_path": str(request.scene_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Scene %s ingested: %d primitives, dim=%d", scene_id, len(scene), features.shape[1])
        return IngestResponse(
            scene_id=scene_id,
            primitive_count=len(scene),
            feature_dim=int(features.shape[1]),
            scene_type=request.scene_type,
            status="ok",
        )

    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _lift_features(scene, request: IngestRequest):
    """
    Extract and project features for a scene.

    Phase 1: uses point cloud colors as a proxy when no image paths are provided.
    Full image-based lifting will be wired in Phase 2.
    """
    import numpy as np
    from python.ingestion.loaders import PointCloud, GaussianSplat

    if isinstance(scene, PointCloud):
        # Phase 1 fallback: use RGB colors as crude 3-dim features,
        # padded/projected to CLIP dim via a random projection.
        # Replace with image-based lifting in Phase 2.
        rng = np.random.default_rng(seed=42)
        D = 512
        proj = rng.standard_normal((3, D)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        raw = scene.colors @ proj  # (N, D)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (raw / norms).astype(np.float32)

    elif isinstance(scene, GaussianSplat):
        rng = np.random.default_rng(seed=42)
        D = 512
        N = len(scene)
        features = rng.standard_normal((N, D)).astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norms

    raise IngestionError(f"Unsupported scene type: {type(scene)}")


def get_scene_registry() -> dict[str, dict]:
    return _scene_registry
