"""POST /api/v1/ingest — scene ingestion endpoint."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from python.api.schemas import IngestRequest, IngestResponse
from python.feature_lifting.clip_extractor import CLIPExtractor, ImageFeatures
from python.feature_lifting.feature_projector import CameraPose, ProjectorFactory
from python.feature_lifting.grounding_dino import GroundingDINODetector
from python.feature_lifting.sam_lifter import SAMLifter, SAMMask
from python.feature_store.index import FeatureIndex
from python.feature_store.persistence import IndexPersistence
from python.ingestion.loaders import PointCloud, Scene, SceneLoaderFactory
from python.ingestion.validators import SceneValidator
from python.utils.errors import IngestionError, ValidationError
from python.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SceneRegistry:
    """Thread-safe async scene registry backed by an asyncio.Lock."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, object]] = {}
        self._lock = asyncio.Lock()

    async def get(self, scene_id: str) -> dict[str, object] | None:
        async with self._lock:
            return self._data.get(scene_id)

    async def set(self, scene_id: str, meta: dict[str, object]) -> None:
        async with self._lock:
            self._data[scene_id] = meta

    async def delete(self, scene_id: str) -> None:
        async with self._lock:
            del self._data[scene_id]

    async def contains(self, scene_id: str) -> bool:
        async with self._lock:
            return scene_id in self._data


_registry = SceneRegistry()


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

        # 4. Build and persist index (positions stored alongside FAISS index)
        from python.api.app import get_persistence
        persistence: IndexPersistence = get_persistence()

        positions = scene.points if isinstance(scene, PointCloud) else scene.means
        feature_index = FeatureIndex(scene_id)
        feature_index.build(features, positions)
        persistence.save(feature_index)

        # 5. Register scene metadata
        await _registry.set(scene_id, {
            "scene_id": scene_id,
            "scene_type": request.scene_type,
            "primitive_count": len(scene),
            "feature_dim": features.shape[1],
            "source_path": str(request.scene_path),
            "created_at": datetime.now(UTC).isoformat(),
        })

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

    if request.config.use_sam:
        features_2d = _refine_with_sam(features_2d, image_paths)

    if request.config.grounding_dino_prompts:
        features_2d = _refine_with_grounding_dino(features_2d, image_paths, request.config.grounding_dino_prompts)

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


def _refine_with_sam(
    features_2d: list[ImageFeatures],
    image_paths: list[Path],
) -> list[ImageFeatures]:
    """
    Refine per-image tile features with SAM segment boundaries.

    Loads each source image, runs SAM automatic mask generation, then replaces
    each tile's feature with the average of all tiles sharing the same segment.
    Falls back to the original features for any image that fails.

    Args:
        features_2d: List of ImageFeatures from CLIPExtractor.extract().
        image_paths: Corresponding source image paths (same order).

    Returns:
        New list of ImageFeatures with SAM-refined embeddings.
    """
    from PIL import Image as PILImage

    lifter = SAMLifter()
    refined: list[ImageFeatures] = []

    for img_path, img_feats in zip(image_paths, features_2d, strict=True):
        try:
            image_rgb = np.array(PILImage.open(img_path).convert("RGB"))
            masks = lifter.segment_image(image_rgb)
            refined.append(lifter.refine_image_features(img_feats, masks))
            logger.debug("SAM refined %s: %d masks applied", img_path.name, len(masks))
        except Exception as exc:
            logger.warning("SAM refinement failed for %s, using raw CLIP features: %s", img_path.name, exc)
            refined.append(img_feats)

    return refined


def _bbox_xyxy_to_xywh(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def _refine_with_grounding_dino(
    features_2d: list[ImageFeatures],
    image_paths: list[Path],
    prompts: list[str],
) -> list[ImageFeatures]:
    """
    Refine per-image tile features by masking to Grounding DINO detections.

    For each image, Grounding DINO is run with the joined prompt caption to
    localise coarse regions. SAM then refines each bounding box into a precise
    mask. Only tiles that fall inside at least one detected region are updated;
    tiles outside all masks retain their original CLIP features.

    This is most useful for cluttered scenes where dense lifting would mix
    foreground object features with irrelevant background regions.

    Args:
        features_2d: Per-image CLIP tile embeddings from CLIPExtractor.
        image_paths: Source image paths in the same order as features_2d.
        prompts: Text descriptions of target objects, e.g. ["chair", "table"].
                 Multiple prompts are joined with " . " (DINO caption format).

    Returns:
        New list of ImageFeatures with DINO+SAM-masked embeddings.
    """
    from PIL import Image as PILImage

    caption = " . ".join(prompts)
    detector = GroundingDINODetector()
    lifter = SAMLifter()
    refined: list[ImageFeatures] = []

    for img_path, img_feats in zip(image_paths, features_2d, strict=True):
        try:
            image_rgb = np.array(PILImage.open(img_path).convert("RGB"))
            detections = detector.detect(image_rgb, caption)
            if not detections:
                logger.debug("Grounding DINO found no objects in %s for caption '%s'", img_path.name, caption)
                refined.append(img_feats)
                continue

            detections = detector.refine_with_sam(image_rgb, detections)
            sam_masks = [
                SAMMask(
                    mask=det.mask,
                    area=int(det.mask.sum()),
                    bbox=_bbox_xyxy_to_xywh(det.bbox_xyxy),
                    score=det.score,
                )
                for det in detections
                if det.mask is not None
            ]
            if sam_masks:
                refined.append(lifter.refine_image_features(img_feats, sam_masks))
                logger.debug(
                    "DINO+SAM refined %s: %d detections applied", img_path.name, len(sam_masks)
                )
            else:
                refined.append(img_feats)
        except Exception as exc:
            logger.warning(
                "DINO+SAM refinement failed for %s, using raw CLIP features: %s", img_path.name, exc
            )
            refined.append(img_feats)

    return refined


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


def get_scene_registry() -> SceneRegistry:
    return _registry
