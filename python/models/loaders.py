"""Model factory and individual model loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from python.utils.errors import ModelLoadError
from python.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModelLoader(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load(self) -> Any:
        ...


class CLIPLoader(BaseModelLoader):
    """Loads OpenCLIP model and preprocessing transforms."""

    def load(self) -> Any:
        try:
            import open_clip

            model_name = self.config.get("model_name", "ViT-B-32")
            pretrained = self.config.get("pretrained", "openai")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model = model.to(self.device).eval()
            tokenizer = open_clip.get_tokenizer(model_name)
            logger.info("CLIP loaded: %s/%s on %s", model_name, pretrained, self.device)
            return {"model": model, "preprocess": preprocess, "tokenizer": tokenizer}
        except Exception as exc:
            raise ModelLoadError(f"Failed to load CLIP: {exc}") from exc


class SAMLoader(BaseModelLoader):
    """Loads Segment Anything Model."""

    def load(self) -> Any:
        try:
            from segment_anything import SamPredictor, sam_model_registry

            checkpoint = self.config.get("checkpoint", "models/weights/sam_vit_h_4b8939.pth")
            model_type = self.config.get("model_type", "vit_h")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam = sam.to(self.device)
            predictor = SamPredictor(sam)
            logger.info("SAM loaded: %s on %s", model_type, self.device)
            return predictor
        except Exception as exc:
            raise ModelLoadError(f"Failed to load SAM: {exc}") from exc


class GroundingDINOLoader(BaseModelLoader):
    """Loads Grounding DINO for open-vocabulary detection."""

    def load(self) -> Any:
        try:
            # groundingdino-py interface
            from groundingdino.util.inference import load_model

            config_path = self.config.get(
                "config_path", "groundingdino/config/GroundingDINO_SwinB.py"
            )
            checkpoint = self.config.get("checkpoint", "groundingdino_swinb_cogcoor.pth")
            model = load_model(config_path, checkpoint)
            logger.info("Grounding DINO loaded on %s", self.device)
            return model
        except Exception as exc:
            raise ModelLoadError(f"Failed to load Grounding DINO: {exc}") from exc


class DepthAnythingLoader(BaseModelLoader):
    """Loads DepthAnything V2 for monocular depth estimation."""

    def load(self) -> Any:
        try:
            import torch.hub

            model = torch.hub.load(
                "LiheYoung/Depth-Anything",
                "DepthAnything",
                encoder=self.config.get("encoder", "vitl"),
            )
            model = model.to(self.device).eval()
            logger.info("DepthAnything loaded on %s", self.device)
            return model
        except Exception as exc:
            raise ModelLoadError(f"Failed to load DepthAnything: {exc}") from exc


class ModelFactory:
    """
    Factory for creating model loaders by name.

    To add a new model:
        1. Implement a BaseModelLoader subclass
        2. Register it here
    """

    _registry: dict[str, type[BaseModelLoader]] = {
        "clip": CLIPLoader,
        "sam": SAMLoader,
        "grounding_dino": GroundingDINOLoader,
        "depth_anything": DepthAnythingLoader,
    }

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> Any:
        """
        Instantiate and load a model by name.

        Args:
            name: Registered model name.
            config: Optional override config dict.

        Raises:
            ModelLoadError: If name is unknown or loading fails.
        """
        if name not in cls._registry:
            raise ModelLoadError(
                f"Unknown model '{name}'. Available: {list(cls._registry)}"
            )
        loader_cls = cls._registry[name]
        return loader_cls(config or {}).load()

    @classmethod
    def register(cls, name: str, loader_cls: type[BaseModelLoader]) -> None:
        """Register a custom model loader."""
        cls._registry[name] = loader_cls
