"""Dense CLIP feature extraction from images via sliding-window tiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from python.models.registry import ModelRegistry
from python.utils.errors import FeatureLiftingError
from python.utils.logging import get_logger

logger = get_logger(__name__)

TILE_SIZE = 224
TILE_STRIDE = 112  # 50% overlap
BATCH_SIZE = 32


@dataclass
class ImageFeatures:
    """Dense CLIP patch features for a single image."""

    embeddings: np.ndarray   # (H_tiles, W_tiles, D) float32, L2-normalized
    image_hw: tuple[int, int]  # original image (H, W)
    tile_size: int
    tile_stride: int


class CLIPExtractor:
    """
    Extract dense CLIP patch features from images via sliding-window tiling.

    Each image is split into overlapping tiles; each tile is encoded by CLIP's
    visual encoder. The result is a spatial grid of features that preserves
    rough spatial correspondence with the original image.

    Usage:
        extractor = CLIPExtractor()
        features = extractor.extract(image_paths)  # list of ImageFeatures
    """

    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        tile_stride: int = TILE_STRIDE,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.batch_size = batch_size
        self._registry = ModelRegistry()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _clip(self) -> dict:
        return self._registry.get("clip")  # type: ignore[return-value]

    @property
    def feature_dim(self) -> int:
        """Output embedding dimension (512 for ViT-B/32, 768 for ViT-L/14)."""
        model = self._clip["model"]
        return model.visual.output_dim  # type: ignore[attr-defined]

    def extract(self, image_paths: list[Path]) -> list[ImageFeatures]:
        """
        Extract dense features for a list of images.

        Args:
            image_paths: Paths to input images (JPEG, PNG, etc.).

        Returns:
            One ImageFeatures per input image.
        """
        results = []
        for path in image_paths:
            try:
                features = self._extract_single(path)
                results.append(features)
            except Exception as exc:
                raise FeatureLiftingError(f"CLIP extraction failed for {path}: {exc}") from exc
        return results

    def _extract_single(self, path: Path) -> ImageFeatures:
        image = Image.open(path).convert("RGB")
        H, W = image.height, image.width

        tiles, positions = self._tile_image(image)

        # Batch inference
        embeddings = self._encode_tiles(tiles)  # (N_tiles, D)

        # Arrange back into spatial grid
        n_h = (H - self.tile_size) // self.tile_stride + 1
        n_w = (W - self.tile_size) // self.tile_stride + 1
        grid = embeddings.reshape(n_h, n_w, -1)

        return ImageFeatures(
            embeddings=grid,
            image_hw=(H, W),
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )

    def _tile_image(self, image: Image.Image) -> tuple[list[Image.Image], list[tuple[int, int]]]:
        """Slice image into overlapping tiles, returning tiles and their (row, col) positions."""
        W, H = image.size
        tiles, positions = [], []
        for r in range(0, H - self.tile_size + 1, self.tile_stride):
            for c in range(0, W - self.tile_size + 1, self.tile_stride):
                tile = image.crop((c, r, c + self.tile_size, r + self.tile_size))
                tiles.append(tile)
                positions.append((r, c))
        return tiles, positions

    def _encode_tiles(self, tiles: list[Image.Image]) -> np.ndarray:
        """Run CLIP visual encoder on batches of tiles."""
        clip = self._clip
        preprocess = clip["preprocess"]
        model = clip["model"]

        all_embeddings = []
        for i in range(0, len(tiles), self.batch_size):
            batch = tiles[i : i + self.batch_size]
            tensor = torch.stack([preprocess(t) for t in batch]).to(self._device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device.type == "cuda"):
                features = model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode text prompts into CLIP embeddings.

        Args:
            texts: List of text strings.

        Returns:
            (N, D) float32 array of L2-normalized text embeddings.
        """
        clip = self._clip
        tokenizer = clip["tokenizer"]
        model = clip["model"]

        tokens = tokenizer(texts).to(self._device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device.type == "cuda"):
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)
