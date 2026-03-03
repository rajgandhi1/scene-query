"""Text query encoding via CLIP."""

from __future__ import annotations

import numpy as np

from python.models.registry import ModelRegistry
from python.utils.errors import QueryError
from python.utils.logging import get_logger

logger = get_logger(__name__)


class CLIPTextEncoder:
    """
    Encode text queries into CLIP embedding space.

    Uses the same CLIP model as feature extraction to ensure text and image
    embeddings are comparable via cosine similarity.

    Usage:
        encoder = CLIPTextEncoder()
        embedding = encoder.encode("red chair")  # (512,) float32
    """

    def __init__(self) -> None:
        self._registry = ModelRegistry()

    def encode(self, query: str) -> np.ndarray:
        """
        Encode a single text query.

        Args:
            query: Natural language query string.

        Returns:
            (D,) float32 L2-normalized embedding.

        Raises:
            QueryError: If encoding fails.
        """
        result = self.encode_batch([query])
        return result[0]

    def encode_batch(self, queries: list[str]) -> np.ndarray:
        """
        Encode multiple queries in one forward pass.

        Args:
            queries: List of query strings.

        Returns:
            (N, D) float32 L2-normalized embeddings.
        """
        try:
            import torch

            clip = self._registry.get("clip")
            tokenizer = clip["tokenizer"]
            model = clip["model"]
            device = next(model.parameters()).device

            tokens = tokenizer(queries).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                embeddings = model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            result = embeddings.cpu().numpy().astype(np.float32)
            logger.debug("Encoded %d queries, dim=%d", len(queries), result.shape[1])
            return result
        except Exception as exc:
            raise QueryError(f"Text encoding failed: {exc}") from exc
