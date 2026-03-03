"""Grounding DINO integration for open-vocabulary object detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from python.models.registry import ModelRegistry
from python.utils.errors import FeatureLiftingError
from python.utils.logging import get_logger

logger = get_logger(__name__)

# Detection thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


@dataclass
class Detection:
    """A single open-vocabulary detection result."""

    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    mask: np.ndarray | None = None  # (H, W) bool — set after SAM refinement


class GroundingDINODetector:
    """
    Detect objects in images using Grounding DINO.

    In the feature lifting pipeline, Grounding DINO is used to localize
    specific objects before passing bounding-box prompts to SAM for precise
    masks. This produces better masks than SAM's automatic mode for targeted
    queries.

    Workflow:
        1. Run Grounding DINO with a text caption to get bounding boxes.
        2. Pass boxes to SAM predictor to get per-object masks.
        3. Use masks in SAM-guided feature lifting.
    """

    def __init__(self) -> None:
        self._registry = ModelRegistry()

    def detect(
        self,
        image_rgb: np.ndarray,
        caption: str,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> list[Detection]:
        """
        Run open-vocabulary detection on an image.

        Args:
            image_rgb: (H, W, 3) uint8 image.
            caption: Text description of objects to detect (e.g. "chair . table").
                     Use " . " as separator for multiple classes.
            box_threshold: Minimum box confidence score.
            text_threshold: Minimum text-box alignment score.

        Returns:
            List of Detection objects.
        """
        try:
            from groundingdino.util.inference import predict
            from PIL import Image
            import torch

            model = self._registry.get("grounding_dino")
            pil_image = Image.fromarray(image_rgb)

            # groundingdino-py returns boxes in (cx, cy, w, h) normalized coords
            boxes, logits, phrases = predict(
                model=model,
                image=pil_image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            H, W = image_rgb.shape[:2]
            detections = []
            for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
                cx, cy, bw, bh = box
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                detections.append(
                    Detection(
                        label=phrase,
                        score=logit,
                        bbox_xyxy=(x1, y1, x2, y2),
                    )
                )

            logger.debug("Grounding DINO found %d objects for caption '%s'", len(detections), caption)
            return detections

        except Exception as exc:
            raise FeatureLiftingError(f"Grounding DINO detection failed: {exc}") from exc

    def refine_with_sam(
        self,
        image_rgb: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        """
        Refine bounding boxes into precise masks using SAM.

        Args:
            image_rgb: (H, W, 3) uint8 image.
            detections: Detections from Grounding DINO.

        Returns:
            Same detections with .mask field populated.
        """
        try:
            import torch

            predictor = self._registry.get("sam")
            predictor.set_image(image_rgb)

            for det in detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                box = np.array([[x1, y1, x2, y2]])
                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=False,
                )
                det.mask = masks[0]  # (H, W) bool

            return detections
        except Exception as exc:
            raise FeatureLiftingError(f"SAM mask refinement failed: {exc}") from exc
