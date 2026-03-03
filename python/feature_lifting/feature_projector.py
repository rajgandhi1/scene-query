"""Strategy-based feature projection from 2D image space onto 3D primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from python.feature_lifting.clip_extractor import ImageFeatures
from python.ingestion.loaders import GaussianSplat, PointCloud, Scene
from python.utils.errors import FeatureLiftingError
from python.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CameraPose:
    """Camera extrinsics and intrinsics for a single view."""

    R: np.ndarray        # (3, 3) rotation matrix (world → camera)
    t: np.ndarray        # (3,)   translation (world → camera)
    fx: float            # focal length x (pixels)
    fy: float            # focal length y (pixels)
    cx: float            # principal point x
    cy: float            # principal point y
    width: int
    height: int

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project world-space points into this camera's image plane.

        Args:
            points_world: (N, 3) float32 XYZ in world space.

        Returns:
            uvs: (N, 2) float32 pixel coordinates (u=col, v=row).
            valid: (N,) bool mask — True if point projects inside image bounds.
        """
        # World → camera
        pts_cam = (self.R @ points_world.T).T + self.t  # (N, 3)

        # Behind camera
        in_front = pts_cam[:, 2] > 0

        # Perspective divide
        u = self.fx * pts_cam[:, 0] / np.where(pts_cam[:, 2] != 0, pts_cam[:, 2], 1e-8) + self.cx
        v = self.fy * pts_cam[:, 1] / np.where(pts_cam[:, 2] != 0, pts_cam[:, 2], 1e-8) + self.cy

        in_bounds = (
            in_front
            & (u >= 0) & (u < self.width)
            & (v >= 0) & (v < self.height)
        )
        return np.stack([u, v], axis=1), in_bounds


def _sample_features_at_uv(
    image_features: ImageFeatures, uvs: np.ndarray
) -> np.ndarray:
    """
    Sample the feature grid at given pixel coordinates via nearest-tile lookup.

    Args:
        image_features: Dense tile features for one view.
        uvs: (N, 2) float32 pixel (u=col, v=row) coordinates.

    Returns:
        (N, D) float32 feature vectors.
    """
    stride = image_features.tile_stride

    # Convert pixel → tile indices
    tile_cols = np.clip(uvs[:, 0].astype(int) // stride, 0, image_features.embeddings.shape[1] - 1)
    tile_rows = np.clip(uvs[:, 1].astype(int) // stride, 0, image_features.embeddings.shape[0] - 1)

    return image_features.embeddings[tile_rows, tile_cols]  # (N, D)


class FeatureProjector(ABC):
    """Abstract base for projecting 2D image features onto 3D scene primitives."""

    @abstractmethod
    def project(
        self,
        features_2d: list[ImageFeatures],
        camera_poses: list[CameraPose],
        scene: Scene,
        aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Project image features onto 3D primitives.

        Args:
            features_2d: Per-view dense CLIP features.
            camera_poses: One pose per view, same order as features_2d.
            scene: Target 3D scene.
            aggregation: "mean" or "max" over visible views.

        Returns:
            (N, D) float32 per-primitive feature matrix, L2-normalized.
            Primitives visible from no camera receive zero vectors.
        """
        ...


class PointCloudProjector(FeatureProjector):
    """
    Project features onto point cloud primitives.

    For each point, finds all cameras from which it is visible, samples
    the CLIP feature at the corresponding 2D projection, and aggregates.
    """

    def project(
        self,
        features_2d: list[ImageFeatures],
        camera_poses: list[CameraPose],
        scene: Scene,
        aggregation: str = "mean",
    ) -> np.ndarray:
        if not isinstance(scene, PointCloud):
            raise FeatureLiftingError("PointCloudProjector requires a PointCloud scene")

        points = scene.points  # (N, 3)
        N = len(points)
        D = features_2d[0].embeddings.shape[-1]

        accum = np.zeros((N, D), dtype=np.float64)
        counts = np.zeros(N, dtype=np.int32)

        for cam_pose, img_feats in zip(camera_poses, features_2d):
            uvs, valid = cam_pose.project(points)
            if not valid.any():
                continue

            sampled = _sample_features_at_uv(img_feats, uvs[valid])

            if aggregation == "max":
                # Take element-wise max over views
                current = accum[valid]
                accum[valid] = np.maximum(current, sampled.astype(np.float64))
            else:
                accum[valid] += sampled.astype(np.float64)

            counts[valid] += 1

        # Average (skip max — already done element-wise)
        if aggregation == "mean":
            visible = counts > 0
            accum[visible] /= counts[visible, np.newaxis]

        result = accum.astype(np.float32)

        # L2-normalize non-zero vectors
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        result[nonzero] /= norms[nonzero]

        invisible_count = (counts == 0).sum()
        if invisible_count > 0:
            logger.debug("%d points had no visible cameras — zero features assigned", invisible_count)

        return result


class GaussianSplatProjector(FeatureProjector):
    """
    Project features onto Gaussian Splat centers.

    Projects onto Gaussian means (centers) using the same camera projection
    as point clouds. Alpha-compositing aware weighting is left for Phase 3.
    """

    def project(
        self,
        features_2d: list[ImageFeatures],
        camera_poses: list[CameraPose],
        scene: Scene,
        aggregation: str = "mean",
    ) -> np.ndarray:
        if not isinstance(scene, GaussianSplat):
            raise FeatureLiftingError("GaussianSplatProjector requires a GaussianSplat scene")

        # Treat Gaussian centers as points for Phase 1
        point_scene = PointCloud(
            points=scene.means,
            colors=np.zeros_like(scene.means),
        )
        pc_projector = PointCloudProjector()
        return pc_projector.project(features_2d, camera_poses, point_scene, aggregation)


class ProjectorFactory:
    """Return the correct projector for a given scene type."""

    _registry: dict[str, type[FeatureProjector]] = {
        "point_cloud": PointCloudProjector,
        "gaussian_splat": GaussianSplatProjector,
    }

    @classmethod
    def get(cls, scene_type: str) -> FeatureProjector:
        if scene_type not in cls._registry:
            raise FeatureLiftingError(
                f"No projector for scene_type '{scene_type}'. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[scene_type]()
