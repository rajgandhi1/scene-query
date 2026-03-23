"""Integration test: full ingestion on a synthetic scene."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from python.feature_lifting.clip_extractor import ImageFeatures
from python.feature_store.index import FeatureIndex


@pytest.mark.integration
def test_feature_index_round_trip(random_features):
    """Build an index from features and verify search returns the correct top-1."""
    pytest.importorskip("faiss")

    index = FeatureIndex("integration-test")
    index.build(random_features)

    query = random_features[42]
    ids, scores = index.search(query, top_k=1)
    assert ids[0] == 42, f"Top-1 should be query itself, got {ids[0]}"
    assert scores[0] > 0.99


def _make_image_features(n_h: int = 4, n_w: int = 4, D: int = 512) -> ImageFeatures:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n_h, n_w, D)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb /= np.where(norms == 0, 1.0, norms)
    return ImageFeatures(
        embeddings=emb,
        image_hw=(224, 224),
        tile_size=112,
        tile_stride=56,
    )


@pytest.mark.integration
def test_clip_lifting_produces_correct_shape(tiny_point_cloud, tmp_path):
    """CLIP feature lifting produces (N, D) float32 output with correct dim from extractor."""
    from python.api.routes.ingest import _lift_features
    from python.api.schemas import CameraPoseInput, IngestRequest

    # Minimal request: image_dir + one image + matching pose
    img_file = tmp_path / "frame_000.jpg"
    img_file.write_bytes(b"")  # content doesn't matter — CLIPExtractor is mocked

    pose = CameraPoseInput(
        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t=[0.0, 0.0, 3.0],
        fx=112.0, fy=112.0,
        cx=112.0, cy=112.0,
        width=224, height=224,
    )
    request = IngestRequest.model_construct(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=[pose],
        config=IngestRequest.model_fields["config"].default_factory(),
    )

    mock_features = [_make_image_features(D=512)]
    with patch(
        "python.api.routes.ingest.CLIPExtractor.extract", return_value=mock_features
    ):
        features = _lift_features(tiny_point_cloud, request)

    N = len(tiny_point_cloud)
    assert features.shape == (N, 512), f"Expected ({N}, 512), got {features.shape}"
    assert features.dtype == np.float32

    # Visible points must be unit-normalized
    norms = np.linalg.norm(features, axis=1)
    visible = norms > 0
    if visible.any():
        np.testing.assert_allclose(norms[visible], 1.0, atol=1e-5)


@pytest.mark.integration
def test_clip_lifting_errors_without_image_dir(tiny_point_cloud, tmp_path):
    """_lift_features raises IngestionError when image_dir is not provided."""
    from python.api.routes.ingest import _lift_features
    from python.api.schemas import IngestRequest
    from python.utils.errors import IngestionError

    request = IngestRequest.model_construct(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=None,
        camera_poses=None,
        config=IngestRequest.model_fields["config"].default_factory(),
    )

    with pytest.raises(IngestionError, match="image_dir is required"):
        _lift_features(tiny_point_cloud, request)


@pytest.mark.integration
def test_clip_lifting_synthesizes_poses_when_absent(tiny_point_cloud, tmp_path):
    """When camera_poses is None, synthetic orbital poses are generated without error."""
    from python.api.routes.ingest import _lift_features
    from python.api.schemas import IngestRequest

    img_file = tmp_path / "frame_000.jpg"
    img_file.write_bytes(b"")

    request = IngestRequest.model_construct(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=None,
        config=IngestRequest.model_fields["config"].default_factory(),
    )

    mock_features = [_make_image_features(D=512)]
    with patch(
        "python.api.routes.ingest.CLIPExtractor.extract", return_value=mock_features
    ):
        features = _lift_features(tiny_point_cloud, request)

    assert features.shape[0] == len(tiny_point_cloud)
    assert features.shape[1] == 512


@pytest.mark.integration
def test_clip_lifting_with_grounding_dino(tiny_point_cloud, tmp_path):
    """DINO+SAM refinement path produces the same shape as plain CLIP lifting."""
    from unittest.mock import MagicMock, patch

    from python.api.routes.ingest import _lift_features
    from python.api.schemas import CameraPoseInput, IngestionConfig, IngestRequest
    from python.feature_lifting.grounding_dino import Detection
    from python.feature_lifting.sam_lifter import SAMMask

    from PIL import Image as PILImage

    img_file = tmp_path / "frame_000.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(img_file)

    pose = CameraPoseInput(
        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t=[0.0, 0.0, 3.0],
        fx=112.0, fy=112.0,
        cx=112.0, cy=112.0,
        width=224, height=224,
    )
    config = IngestionConfig(grounding_dino_prompts=["chair", "table"])
    request = IngestRequest.model_construct(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=[pose],
        config=config,
    )

    mock_clip_features = [_make_image_features(D=512)]

    # Synthetic DINO detection covering the top-left quadrant
    H, W = 224, 224
    mask_array = np.zeros((H, W), dtype=bool)
    mask_array[:112, :112] = True

    mock_detection = Detection(
        label="chair",
        score=0.85,
        bbox_xyxy=(0.0, 0.0, 112.0, 112.0),
        mask=mask_array,
    )

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [mock_detection]
    mock_detector.refine_with_sam.return_value = [mock_detection]

    mock_lifter = MagicMock()
    mock_lifter.refine_image_features.side_effect = lambda feats, masks: feats

    with (
        patch("python.api.routes.ingest.CLIPExtractor.extract", return_value=mock_clip_features),
        patch("python.api.routes.ingest.GroundingDINODetector", return_value=mock_detector),
        patch("python.api.routes.ingest.SAMLifter", return_value=mock_lifter),
    ):
        features = _lift_features(tiny_point_cloud, request)

    assert features.shape == (len(tiny_point_cloud), 512)
    assert features.dtype == np.float32
    mock_detector.detect.assert_called_once()
    # Verify the caption joins prompts with " . "
    call_args = mock_detector.detect.call_args
    assert call_args[0][1] == "chair . table"
    mock_detector.refine_with_sam.assert_called_once()
    mock_lifter.refine_image_features.assert_called_once()


@pytest.mark.integration
def test_clip_lifting_with_dino_no_detections(tiny_point_cloud, tmp_path):
    """When DINO finds no objects, features fall back to raw CLIP output unchanged."""
    from unittest.mock import MagicMock, patch

    from python.api.routes.ingest import _lift_features
    from python.api.schemas import CameraPoseInput, IngestionConfig, IngestRequest

    from PIL import Image as PILImage

    img_file = tmp_path / "frame_000.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(img_file)

    pose = CameraPoseInput(
        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t=[0.0, 0.0, 3.0],
        fx=112.0, fy=112.0,
        cx=112.0, cy=112.0,
        width=224, height=224,
    )
    config = IngestionConfig(grounding_dino_prompts=["nonexistent_object"])
    request = IngestRequest.model_construct(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=[pose],
        config=config,
    )

    mock_clip_features = [_make_image_features(D=512)]

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []  # nothing found

    mock_lifter = MagicMock()

    with (
        patch("python.api.routes.ingest.CLIPExtractor.extract", return_value=mock_clip_features),
        patch("python.api.routes.ingest.GroundingDINODetector", return_value=mock_detector),
        patch("python.api.routes.ingest.SAMLifter", return_value=mock_lifter),
    ):
        features = _lift_features(tiny_point_cloud, request)

    assert features.shape == (len(tiny_point_cloud), 512)
    # refine_image_features must NOT be called when there are no detections
    mock_lifter.refine_image_features.assert_not_called()


@pytest.mark.integration
def test_clip_lifting_with_sam_refinement(tiny_point_cloud, tmp_path):
    """SAM refinement path produces the same shape as plain CLIP lifting."""
    from unittest.mock import MagicMock

    from python.api.routes.ingest import _lift_features
    from python.api.schemas import CameraPoseInput, IngestionConfig, IngestRequest
    from python.feature_lifting.sam_lifter import SAMMask

    from PIL import Image as PILImage

    img_file = tmp_path / "frame_000.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(img_file)

    pose = CameraPoseInput(
        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t=[0.0, 0.0, 3.0],
        fx=112.0, fy=112.0,
        cx=112.0, cy=112.0,
        width=224, height=224,
    )
    config = IngestionConfig(use_sam=True)
    request = IngestRequest.model_construct(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=[pose],
        config=config,
    )

    mock_clip_features = [_make_image_features(D=512)]

    # Synthetic SAM mask covering the top-left 2×2 tiles
    H, W = 224, 224
    mask_array = np.zeros((H, W), dtype=bool)
    mask_array[:112, :112] = True
    mock_masks = [SAMMask(mask=mask_array, area=int(mask_array.sum()), bbox=(0, 0, 112, 112), score=0.95)]

    mock_sam = MagicMock()
    mock_sam.segment_image.return_value = mock_masks
    mock_sam.refine_image_features.side_effect = lambda feats, masks: feats  # pass-through

    with (
        patch("python.api.routes.ingest.CLIPExtractor.extract", return_value=mock_clip_features),
        patch("python.api.routes.ingest.SAMLifter", return_value=mock_sam),
    ):
        features = _lift_features(tiny_point_cloud, request)

    assert features.shape == (len(tiny_point_cloud), 512)
    assert features.dtype == np.float32
    # SAMLifter methods must have been called once per image
    mock_sam.segment_image.assert_called_once()
    mock_sam.refine_image_features.assert_called_once()
