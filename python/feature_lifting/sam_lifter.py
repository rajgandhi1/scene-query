"""SAM-guided feature lifting for clean object-level feature boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from python.feature_lifting.clip_extractor import ImageFeatures
from python.feature_lifting.feature_projector import CameraPose, PointCloudProjector
from python.ingestion.loaders import PointCloud
from python.models.registry import ModelRegistry
from python.utils.errors import FeatureLiftingError
from python.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SAMMask:
    """A single SAM segmentation mask for one image."""

    mask: np.ndarray         # (H, W) bool
    area: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float


class SAMLifter:
    """
    Refine feature boundaries using SAM segmentation masks.

    Instead of assigning per-tile features independently, SAM-guided lifting
    ensures that all points projecting into the same segment receive the
    same averaged feature — giving cleaner object-level boundaries.

    This is the primary (higher-quality) lifting path. Falls back to
    PointCloudProjector on failure.
    """

    def __init__(self) -> None:
        self._registry = ModelRegistry()

    def segment_image(self, image_rgb: np.ndarray) -> list[SAMMask]:
        """
        Run SAM automatic mask generation on a single image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            List of SAMMask objects sorted by area descending.
        """
        try:
            from segment_anything import SamAutomaticMaskGenerator

            sam = self._registry.get("sam")
            generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=100,
            )
            raw_masks = generator.generate(image_rgb)
            masks = [
                SAMMask(
                    mask=m["segmentation"],
                    area=m["area"],
                    bbox=tuple(m["bbox"]),  # type: ignore[arg-type]
                    score=m["predicted_iou"],
                )
                for m in raw_masks
            ]
            return sorted(masks, key=lambda m: m.area, reverse=True)
        except Exception as exc:
            raise FeatureLiftingError(f"SAM segmentation failed: {exc}") from exc

    def lift_with_masks(
        self,
        image_features: ImageFeatures,
        masks: list[SAMMask],
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        """
        Produce per-pixel features by averaging tile features within each mask.

        Args:
            image_features: Dense CLIP tile embeddings.
            masks: SAM masks for the same image.
            image_hw: (H, W) of the original image.

        Returns:
            (H, W, D) float32 per-pixel feature map.
        """
        H, W = image_hw
        D = image_features.embeddings.shape[-1]
        stride = image_features.tile_stride

        # Start with per-tile features upsampled to pixel space
        pixel_features = np.zeros((H, W, D), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                tr = min(r // stride, image_features.embeddings.shape[0] - 1)
                tc = min(c // stride, image_features.embeddings.shape[1] - 1)
                pixel_features[r, c] = image_features.embeddings[tr, tc]

        # Refine: within each mask region, replace with mask-averaged feature
        for sam_mask in masks:
            region = sam_mask.mask
            if not region.any():
                continue
            avg_feat = pixel_features[region].mean(axis=0)
            norm = np.linalg.norm(avg_feat)
            if norm > 0:
                avg_feat /= norm
            pixel_features[region] = avg_feat

        return pixel_features
