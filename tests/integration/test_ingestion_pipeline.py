"""Integration test: full ingestion on a synthetic scene."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python.feature_lifting.clip_extractor import ImageFeatures
from python.feature_store.index import FeatureIndex
from python.ingestion.loaders import PointCloud


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
    request = IngestRequest(
        scene_path=tmp_path / "scene.ply",  # won't be read; validator skipped below
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=[pose],
    )
    # Bypass scene_path validator for this unit-level test
    object.__setattr__(request, "scene_path", tmp_path / "scene.ply")

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

    request = IngestRequest(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=None,
    )
    object.__setattr__(request, "scene_path", tmp_path / "scene.ply")

    with pytest.raises(IngestionError, match="image_dir is required"):
        _lift_features(tiny_point_cloud, request)


@pytest.mark.integration
def test_clip_lifting_synthesizes_poses_when_absent(tiny_point_cloud, tmp_path):
    """When camera_poses is None, synthetic orbital poses are generated without error."""
    from python.api.routes.ingest import _lift_features
    from python.api.schemas import IngestRequest

    img_file = tmp_path / "frame_000.jpg"
    img_file.write_bytes(b"")

    request = IngestRequest(
        scene_path=tmp_path / "scene.ply",
        scene_type="point_cloud",
        image_dir=tmp_path,
        camera_poses=None,
    )
    object.__setattr__(request, "scene_path", tmp_path / "scene.ply")

    mock_features = [_make_image_features(D=512)]
    with patch(
        "python.api.routes.ingest.CLIPExtractor.extract", return_value=mock_features
    ):
        features = _lift_features(tiny_point_cloud, request)

    assert features.shape[0] == len(tiny_point_cloud)
    assert features.shape[1] == 512
