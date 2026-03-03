"""Unit tests for FeatureProjector implementations."""

from __future__ import annotations

import numpy as np
import pytest

from python.feature_lifting.feature_projector import CameraPose, PointCloudProjector
from python.ingestion.loaders import PointCloud


def make_camera(
    tx: float = 0.0, ty: float = 0.0, tz: float = 3.0,
    width: int = 64, height: int = 64,
) -> CameraPose:
    """Camera looking down -Z with a simple translation."""
    return CameraPose(
        R=np.eye(3, dtype=np.float32),
        t=np.array([tx, ty, tz], dtype=np.float32),
        fx=50.0, fy=50.0,
        cx=width / 2, cy=height / 2,
        width=width, height=height,
    )


def make_image_features(n_h: int = 4, n_w: int = 4, D: int = 512) -> "ImageFeatures":
    from python.feature_lifting.clip_extractor import ImageFeatures
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n_h, n_w, D)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb /= np.where(norms == 0, 1.0, norms)
    return ImageFeatures(
        embeddings=emb,
        image_hw=(n_h * 56, n_w * 56),  # stride 56 → tile_size 112
        tile_size=112,
        tile_stride=56,
    )


def test_projection_output_shape(tiny_point_cloud):
    projector = PointCloudProjector()
    features_2d = [make_image_features()]
    poses = [make_camera()]

    result = projector.project(features_2d, poses, tiny_point_cloud)

    assert result.shape == (len(tiny_point_cloud), 512), (
        f"Expected ({len(tiny_point_cloud)}, 512), got {result.shape}"
    )
    assert result.dtype == np.float32


def test_visible_features_are_normalized(tiny_point_cloud):
    projector = PointCloudProjector()
    result = projector.project([make_image_features()], [make_camera()], tiny_point_cloud)

    norms = np.linalg.norm(result, axis=1)
    visible = norms > 0
    # All visible point features should be approximately unit norm
    if visible.any():
        np.testing.assert_allclose(norms[visible], 1.0, atol=1e-5)


def test_camera_projection_in_bounds():
    cam = make_camera(tz=3.0, width=64, height=64)
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    uvs, valid = cam.project(points)
    # Point at origin should project near center of camera
    assert valid[0], "Point at origin should be visible"
    assert 0 <= uvs[0, 0] < cam.width
    assert 0 <= uvs[0, 1] < cam.height


def test_behind_camera_points_invalid():
    cam = make_camera(tz=3.0)
    # Point behind camera (z > tz in camera space means behind for this setup)
    points = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    _, valid = cam.project(points)
    assert not valid[0], "Point behind camera should be invalid"
