"""Scene loaders for PLY point clouds, Gaussian splats, and OBJ meshes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from python.utils.errors import IngestionError
from python.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PointCloud:
    """Internal representation of a point cloud scene."""

    points: np.ndarray          # (N, 3) float32 XYZ
    colors: np.ndarray          # (N, 3) float32 RGB in [0, 1]
    normals: np.ndarray | None = None  # (N, 3) float32, optional
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class GaussianSplat:
    """Internal representation of a 3D Gaussian Splat scene."""

    means: np.ndarray           # (N, 3) float32 — Gaussian centers
    scales: np.ndarray          # (N, 3) float32 — log-scale
    rotations: np.ndarray       # (N, 4) float32 — quaternion (w, x, y, z)
    opacities: np.ndarray       # (N,)   float32 — logit opacity
    sh_coeffs: np.ndarray       # (N, K, 3) float32 — spherical harmonics
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.means)


# Union type for any supported scene
Scene = PointCloud | GaussianSplat


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> Scene:
        ...


class PLYLoader(BaseLoader):
    """Load PLY files as PointCloud or GaussianSplat depending on properties."""

    def load(self, path: Path) -> Scene:
        try:
            from plyfile import PlyData

            plydata = PlyData.read(str(path))
            vertex = plydata["vertex"]

            # Gaussian splat PLY has specific properties
            if "scale_0" in vertex.data.dtype.names:
                return self._load_splat(vertex, path)
            return self._load_pointcloud(vertex, path)

        except Exception as exc:
            raise IngestionError(f"Failed to load PLY '{path}': {exc}") from exc

    def _load_pointcloud(self, vertex: Any, path: Path) -> PointCloud:  # type: ignore[misc]
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

        if all(c in vertex.data.dtype.names for c in ("red", "green", "blue")):
            colors = np.stack([
                vertex["red"], vertex["green"], vertex["blue"]
            ], axis=1).astype(np.float32) / 255.0
        else:
            colors = np.ones((len(points), 3), dtype=np.float32)

        normals = None
        if all(n in vertex.data.dtype.names for n in ("nx", "ny", "nz")):
            normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)

        logger.info("Loaded point cloud: %d points from %s", len(points), path.name)
        return PointCloud(points=points, colors=colors, normals=normals, source_path=path)

    def _load_splat(self, vertex: Any, path: Path) -> GaussianSplat:  # type: ignore[misc]
        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        scales = np.stack(
            [vertex[f"scale_{i}"] for i in range(3)], axis=1
        ).astype(np.float32)
        rotations = np.stack(
            [vertex[f"rot_{i}"] for i in range(4)], axis=1
        ).astype(np.float32)
        opacities = np.array(vertex["opacity"], dtype=np.float32)

        # Collect SH coefficients (degree 0 = f_dc, higher = f_rest)
        dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1)
        sh_coeffs = dc[:, np.newaxis, :]  # simplified: only degree 0

        logger.info("Loaded Gaussian splat: %d Gaussians from %s", len(means), path.name)
        return GaussianSplat(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            source_path=path,
        )


class SplatLoader(BaseLoader):
    """Load binary .splat files (Antimatter15 format)."""

    # Each Gaussian: 3×f32 pos + 3×f32 scale + 4×u8 color + 4×u8 quaternion = 32 bytes
    _RECORD_BYTES = 32

    def load(self, path: Path) -> GaussianSplat:
        try:
            data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            n = len(data) // self._RECORD_BYTES
            data = data[: n * self._RECORD_BYTES].reshape(n, self._RECORD_BYTES)

            means = data[:, :12].view(np.float32).reshape(n, 3)
            scales = data[:, 12:24].view(np.float32).reshape(n, 3)
            # color (RGBA u8) and quaternion (u8 packed) — decode as needed
            opacities = data[:, 27].astype(np.float32) / 255.0

            logger.info("Loaded .splat: %d Gaussians from %s", n, path.name)
            return GaussianSplat(
                means=means.copy(),
                scales=scales.copy(),
                rotations=np.zeros((n, 4), dtype=np.float32),
                opacities=opacities,
                sh_coeffs=np.zeros((n, 1, 3), dtype=np.float32),
                source_path=path,
            )
        except Exception as exc:
            raise IngestionError(f"Failed to load .splat '{path}': {exc}") from exc


class SceneLoaderFactory:
    """Dispatch to the correct loader based on file extension."""

    _loaders: dict[str, type[BaseLoader]] = {
        ".ply": PLYLoader,
        ".splat": SplatLoader,
    }

    @classmethod
    def get_loader(cls, path: Path) -> BaseLoader:
        loader_cls = cls._loaders.get(path.suffix)
        if loader_cls is None:
            raise IngestionError(
                f"No loader available for extension '{path.suffix}'. "
                f"Supported: {list(cls._loaders)}"
            )
        return loader_cls()

    @classmethod
    def load(cls, path: Path) -> Scene:
        return cls.get_loader(path).load(path)


# plyfile is optional — provide a helpful error if missing
try:
    from plyfile import PlyData as _  # noqa: F401
except ImportError:
    logger.warning("plyfile not installed — PLY loading unavailable. Run: pip install plyfile")

from typing import Any  # noqa: E402 (needed for type hints above)
