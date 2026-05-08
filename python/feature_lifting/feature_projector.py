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


_ALPHA_CONTRIB_THRESHOLD = 1e-4  # minimum T*alpha to be considered a rendering contributor


class GaussianSplatProjector(FeatureProjector):
    """
    Project features onto Gaussian Splat centers with alpha-compositing aware weighting.

    For each camera view, visible Gaussians are sorted front-to-back by depth. Their
    rendering contribution is computed via the standard alpha compositing formula:
        weight_i = T_i * alpha_i,   T_i = prod_{j<i}(1 - alpha_j)
    Features are accumulated as a weighted sum and then normalized by total weight.

    Opacities stored on ``GaussianSplat`` are expected to be logit values (as produced
    by the standard 3DGS PLY format); sigmoid is applied to recover alpha in [0, 1].
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

        # Logit opacities → alpha in [0, 1] via sigmoid
        alphas = 1.0 / (1.0 + np.exp(-scene.opacities.astype(np.float64)))  # (N,)

        N = len(scene.means)
        D = features_2d[0].embeddings.shape[-1]

        accum = np.zeros((N, D), dtype=np.float64)
        weight_sum = np.zeros(N, dtype=np.float64)

        for cam_pose, img_feats in zip(camera_poses, features_2d):
            uvs, valid = cam_pose.project(scene.means)
            if not valid.any():
                continue

            # Depth of each Gaussian in this camera (z in camera space)
            pts_cam = (cam_pose.R @ scene.means.T).T + cam_pose.t  # (N, 3)
            valid_idx = np.where(valid)[0]

            # Assign each visible Gaussian to its nearest feature tile.
            # Transmittance is computed per-tile so that Gaussians on different
            # rays (different tiles) cannot occlude each other.
            H_tiles = img_feats.embeddings.shape[0]
            W_tiles = img_feats.embeddings.shape[1]
            stride = img_feats.tile_stride
            tile_cols = np.clip(uvs[valid_idx, 0].astype(int) // stride, 0, W_tiles - 1)
            tile_rows = np.clip(uvs[valid_idx, 1].astype(int) // stride, 0, H_tiles - 1)
            tile_ids = tile_rows * W_tiles + tile_cols  # (K,) flat tile index

            for tile_id in np.unique(tile_ids):
                t_mask = tile_ids == tile_id
                t_global_idx = valid_idx[t_mask]  # Gaussians in this tile (global indices)
                t_depths = pts_cam[t_global_idx, 2]

                # Sort front-to-back within this tile
                sort_order = np.argsort(t_depths)
                sorted_idx = t_global_idx[sort_order]
                sorted_alphas = alphas[sorted_idx]

                # T_i = prod_{j<i}(1 - alpha_j)
                T = np.ones(len(sorted_idx), dtype=np.float64)
                if len(sorted_idx) > 1:
                    T[1:] = np.cumprod(1.0 - sorted_alphas[:-1])

                contrib = T * sorted_alphas  # rendering contribution per Gaussian

                # Fetch the single tile feature (all Gaussians in this tile share it)
                tile_row, tile_col = divmod(int(tile_id), W_tiles)
                tile_feat = img_feats.embeddings[tile_row, tile_col].astype(np.float64)  # (D,)

                # Skip Gaussians whose per-view contribution is below threshold
                meaningful = contrib > _ALPHA_CONTRIB_THRESHOLD
                if not meaningful.any():
                    continue

                mi = sorted_idx[meaningful]
                mi_contrib = contrib[meaningful]

                if aggregation == "max":
                    accum[mi] = np.maximum(accum[mi], tile_feat)
                    weight_sum[mi] = np.maximum(weight_sum[mi], mi_contrib)
                else:  # mean
                    accum[mi] += mi_contrib[:, np.newaxis] * tile_feat
                    weight_sum[mi] += mi_contrib

        # Weighted mean: divide accumulated sum by total contribution weight
        if aggregation == "mean":
            has_weight = weight_sum > 0
            accum[has_weight] /= weight_sum[has_weight, np.newaxis]

        result = accum.astype(np.float32)

        # L2-normalize non-zero vectors
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        result[nonzero] /= norms[nonzero]

        invisible_count = (weight_sum == 0).sum()
        if invisible_count > 0:
            logger.debug(
                "%d Gaussians had no rendering contribution — zero features assigned",
                invisible_count,
            )

        return result


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
