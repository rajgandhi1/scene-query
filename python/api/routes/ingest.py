"""POST /api/v1/ingest — scene ingestion endpoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from python.api.schemas import IngestRequest, IngestResponse
from python.feature_lifting.clip_extractor import CLIPExtractor
from python.feature_lifting.feature_projector import CameraPose, ProjectorFactory
from python.feature_store.index import FeatureIndex
from python.feature_store.persistence import IndexPersistence
from python.ingestion.loaders import GaussianSplat, PointCloud, Scene, SceneLoaderFactory
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

        # 3. Feature lifting: CLIP features extracted from images, projected onto 3D primitives
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


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _lift_features(scene: Scene, request: IngestRequest) -> np.ndarray:
    """
    Extract dense CLIP features from source images and project onto 3D primitives.

    Requires ``request.image_dir`` to point to a directory of source images.
    Camera poses are used when provided via ``request.camera_poses``; otherwise
    synthetic orbital cameras are generated around the scene bounding box.

    Returns:
        (N, D) float32 L2-normalized feature matrix, where D is the CLIP model's
        output dimension (e.g. 512 for ViT-B/32, 768 for ViT-L/14).
    """
    if request.image_dir is None:
        raise IngestionError(
            "image_dir is required for CLIP feature lifting. "
            "Provide a directory containing source images for this scene."
        )

    image_dir = Path(request.image_dir)
    if not image_dir.is_dir():
        raise IngestionError(f"image_dir does not exist or is not a directory: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise IngestionError(
            f"No images found in {image_dir}. "
            f"Supported formats: {sorted(_IMAGE_EXTENSIONS)}"
        )

    logger.info("Extracting CLIP features from %d images in %s", len(image_paths), image_dir)
    extractor = CLIPExtractor()
    features_2d = extractor.extract(image_paths)

    if request.camera_poses is not None:
        if len(request.camera_poses) != len(image_paths):
            raise IngestionError(
                f"camera_poses length ({len(request.camera_poses)}) must match "
                f"number of images found ({len(image_paths)})"
            )
        camera_poses = [
            CameraPose(
                R=np.array(cp.R, dtype=np.float32),
                t=np.array(cp.t, dtype=np.float32),
                fx=cp.fx,
                fy=cp.fy,
                cx=cp.cx,
                cy=cp.cy,
                width=cp.width,
                height=cp.height,
            )
            for cp in request.camera_poses
        ]
    else:
        logger.warning(
            "No camera_poses provided — synthesizing %d orbital cameras around scene AABB. "
            "Supply camera_poses for accurate per-view feature correspondence.",
            len(image_paths),
        )
        camera_poses = _synthesize_orbital_poses(
            scene, n_views=len(image_paths), image_hw=features_2d[0].image_hw
        )

    projector = ProjectorFactory.get(request.scene_type)
    features = projector.project(
        features_2d,
        camera_poses,
        scene,
        aggregation=request.config.aggregation,
    )
    logger.info(
        "Feature lifting complete: %d primitives × dim=%d", features.shape[0], features.shape[1]
    )
    return features


def _synthesize_orbital_poses(
    scene: Scene,
    n_views: int,
    image_hw: tuple[int, int],
) -> list[CameraPose]:
    """
    Generate orbital cameras uniformly distributed in azimuth around the scene AABB.

    Each camera is placed at twice the scene radius from its center and oriented
    to look inward. Used as a pose-free fallback when ``camera_poses`` is absent.
    """
    pts = scene.points if isinstance(scene, PointCloud) else scene.means
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max()) * 2.0
    radius = max(radius, 1e-3)

    H, W = image_hw
    fx = fy = float(max(W, H))
    cx, cy = W / 2.0, H / 2.0

    poses: list[CameraPose] = []
    for i in range(n_views):
        azimuth = 2.0 * np.pi * i / n_views
        cam_pos = center + radius * np.array(
            [np.cos(azimuth), 0.0, np.sin(azimuth)], dtype=np.float32
        )

        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(forward, world_up))) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # OpenCV convention: rows are right, down (-up), forward
        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        poses.append(CameraPose(R=R, t=t, fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H))

    return poses


def get_scene_registry() -> dict[str, dict]:
    return _scene_registry
