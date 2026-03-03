"""Preprocessing pipeline: image undistortion and depth estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from python.models.registry import ModelRegistry
from python.utils.errors import IngestionError
from python.utils.events import event_bus
from python.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics + optional distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float] = field(default_factory=list)  # k1, k2, p1, p2[, k3]
    width: int = 0
    height: int = 0


@dataclass
class PreprocessedView:
    """A single preprocessed image with associated depth and intrinsics."""

    image_path: Path
    depth_map: np.ndarray        # (H, W) float32 metric depth in meters
    intrinsics: CameraIntrinsics
    undistorted: bool = False


class UndistortionStep:
    """Undistort images using OpenCV given camera intrinsics."""

    def process(self, image_path: Path, intrinsics: CameraIntrinsics) -> Path:
        """
        Undistort a single image.

        Args:
            image_path: Input distorted image path.
            intrinsics: Camera intrinsics with distortion coefficients.

        Returns:
            Path to the undistorted image (written alongside original).
        """
        if not intrinsics.dist_coeffs:
            return image_path  # Already undistorted or pinhole camera

        try:
            import cv2

            img = cv2.imread(str(image_path))
            K = np.array([
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ], dtype=np.float64)
            dist = np.array(intrinsics.dist_coeffs, dtype=np.float64)
            undistorted = cv2.undistort(img, K, dist)
            out_path = image_path.with_stem(image_path.stem + "_undistorted")
            cv2.imwrite(str(out_path), undistorted)
            return out_path
        except Exception as exc:
            raise IngestionError(f"Undistortion failed for {image_path}: {exc}") from exc


class DepthEstimationStep:
    """Estimate metric depth maps using DepthAnything V2."""

    def __init__(self) -> None:
        self._registry = ModelRegistry()

    def estimate(self, image_path: Path) -> np.ndarray:
        """
        Estimate a depth map for a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            (H, W) float32 depth map in relative scale (not metric without calibration).
        """
        try:
            import torch
            from PIL import Image

            model = self._registry.get("depth_anything")
            image = Image.open(image_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            with torch.no_grad():
                depth = model(img_tensor.unsqueeze(0))

            return depth.squeeze().cpu().numpy().astype(np.float32)
        except Exception as exc:
            raise IngestionError(f"Depth estimation failed for {image_path}: {exc}") from exc


class PreprocessingPipeline:
    """
    Run all preprocessing steps over a set of images.

    Steps:
        1. Undistort images (if intrinsics provided)
        2. Estimate depth maps
    """

    def __init__(
        self,
        run_undistortion: bool = True,
        run_depth: bool = True,
    ) -> None:
        self._undistort = UndistortionStep() if run_undistortion else None
        self._depth = DepthEstimationStep() if run_depth else None

    def run(
        self,
        image_paths: list[Path],
        intrinsics: list[CameraIntrinsics] | None = None,
    ) -> list[PreprocessedView]:
        """
        Preprocess all images.

        Args:
            image_paths: Input image paths.
            intrinsics: Per-image camera intrinsics (optional; skips undistortion if None).

        Returns:
            List of PreprocessedView objects.
        """
        results = []
        total = len(image_paths)

        for i, img_path in enumerate(image_paths):
            event_bus.emit("preprocessing.progress", {"step": i, "total": total})

            cam_intrinsics = intrinsics[i] if intrinsics else None
            processed_path = img_path

            if self._undistort and cam_intrinsics and cam_intrinsics.dist_coeffs:
                processed_path = self._undistort.process(img_path, cam_intrinsics)
                undistorted = True
            else:
                undistorted = False

            depth_map: np.ndarray
            if self._depth:
                depth_map = self._depth.estimate(processed_path)
            else:
                # Placeholder zero depth
                depth_map = np.zeros((1, 1), dtype=np.float32)

            results.append(PreprocessedView(
                image_path=processed_path,
                depth_map=depth_map,
                intrinsics=cam_intrinsics or CameraIntrinsics(fx=1, fy=1, cx=0, cy=0),
                undistorted=undistorted,
            ))
            logger.debug("Preprocessed %d/%d: %s", i + 1, total, img_path.name)

        event_bus.emit("preprocessing.complete", {"total": total})
        return results
