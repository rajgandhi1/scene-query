"""Unit tests for FeatureProjector implementations."""

from __future__ import annotations

import numpy as np
import pytest

from python.feature_lifting.feature_projector import (
    CameraPose,
    GaussianSplatProjector,
    PointCloudProjector,
)
from python.ingestion.loaders import GaussianSplat, PointCloud


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
    # With R=I and t=[0,0,3]: pts_cam = pts_world + t.
    # For cam_z <= 0 (behind camera) we need world_z < -3.
    points = np.array([[0.0, 0.0, -5.0]], dtype=np.float32)
    _, valid = cam.project(points)
    assert not valid[0], "Point behind camera should be invalid"


# ---------------------------------------------------------------------------
# GaussianSplatProjector — alpha-compositing aware feature weighting
# ---------------------------------------------------------------------------

def _make_splat(
    means: np.ndarray,
    logit_opacities: np.ndarray,
) -> GaussianSplat:
    """Minimal GaussianSplat for projection tests."""
    N = len(means)
    return GaussianSplat(
        means=means.astype(np.float32),
        scales=np.zeros((N, 3), dtype=np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
        opacities=logit_opacities.astype(np.float32),
        sh_coeffs=np.zeros((N, 1, 3), dtype=np.float32),
    )


def test_gaussian_splat_projection_output_shape(tiny_gaussian_splat):
    projector = GaussianSplatProjector()
    result = projector.project([make_image_features()], [make_camera()], tiny_gaussian_splat)
    assert result.shape == (len(tiny_gaussian_splat), 512)
    assert result.dtype == np.float32


def test_gaussian_splat_features_are_normalized(tiny_gaussian_splat):
    projector = GaussianSplatProjector()
    result = projector.project([make_image_features()], [make_camera()], tiny_gaussian_splat)
    norms = np.linalg.norm(result, axis=1)
    visible = norms > 0
    if visible.any():
        np.testing.assert_allclose(norms[visible], 1.0, atol=1e-5)


def test_fully_transparent_gaussian_gets_no_features():
    """A Gaussian with logit opacity = -100 (alpha ≈ 0) should accumulate zero weight."""
    cam = make_camera(tz=3.0)
    # One transparent Gaussian at origin, well inside the camera frustum
    means = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    logit_opacities = np.array([-100.0])  # sigmoid(-100) ≈ 3.7e-44
    splat = _make_splat(means, logit_opacities)

    projector = GaussianSplatProjector()
    result = projector.project([make_image_features()], [cam], splat)

    # Near-zero alpha → near-zero contribution → zero feature vector (no L2-norm possible)
    assert np.allclose(result[0], 0.0, atol=1e-6), (
        "Fully transparent Gaussian should receive zero features"
    )


def test_opaque_gaussian_gets_nonzero_features():
    """A Gaussian with logit opacity = +100 (alpha ≈ 1) should receive features."""
    cam = make_camera(tz=3.0)
    means = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    logit_opacities = np.array([100.0])  # sigmoid(100) ≈ 1.0
    splat = _make_splat(means, logit_opacities)

    projector = GaussianSplatProjector()
    result = projector.project([make_image_features()], [cam], splat)

    assert np.linalg.norm(result[0]) > 0.99, (
        "Fully opaque Gaussian should receive L2-normalized features"
    )


def test_opaque_front_occludes_rear_gaussian():
    """
    When a fully opaque Gaussian sits in front of another, the rear one's
    transmittance drops to zero — it should accumulate zero weight.
    """
    cam = make_camera(tz=0.0)  # camera at z=0, looking toward -z (R=I, t=0)
    # front: z = -1 (depth=1 in cam space since t=0), rear: z = -2 (depth=2)
    # With R=I and t=0: cam_z = world_z, so front must have larger world_z
    # Actually CameraPose: pts_cam = R @ pts_world.T + t, with R=I, t=0 → pts_cam = pts_world
    # Camera sees points with pts_cam[:, 2] > 0, so z in world > 0 to be in front
    means = np.array([
        [0.0, 0.0, 2.0],  # front (depth=2, closer)
        [0.0, 0.0, 1.0],  # rear (depth=1, farther) — wait, lower z = closer
    ], dtype=np.float32)
    # front-to-back sort is ascending depth (z in camera space)
    # With t=0, R=I: cam_z = world_z
    # Gaussian at world_z=1 → depth 1 (front), Gaussian at world_z=2 → depth 2 (rear)
    means = np.array([
        [0.0, 0.0, 1.0],  # front (depth=1)
        [0.0, 0.0, 2.0],  # rear  (depth=2)
    ], dtype=np.float32)
    logit_opacities = np.array([100.0, 100.0])  # both fully opaque

    splat = _make_splat(means, logit_opacities)
    projector = GaussianSplatProjector()

    # Use a camera whose frustum covers both points (cam at z=-5 looking +z)
    cam = CameraPose(
        R=np.eye(3, dtype=np.float32),
        t=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        fx=50.0, fy=50.0,
        cx=32.0, cy=32.0,
        width=64, height=64,
    )
    result = projector.project([make_image_features()], [cam], splat)

    # Front Gaussian (index 0) should have a feature; rear (index 1) should be zero
    assert np.linalg.norm(result[0]) > 0.5, "Front opaque Gaussian should have features"
    assert np.allclose(result[1], 0.0, atol=1e-6), (
        "Rear Gaussian fully occluded by opaque front should have zero features"
    )


def test_transparent_front_passes_weight_to_rear():
    """
    When the front Gaussian is fully transparent, the rear Gaussian's
    transmittance stays ≈1 and it receives full feature weight.
    """
    means = np.array([
        [0.0, 0.0, 1.0],  # front (transparent)
        [0.0, 0.0, 2.0],  # rear  (opaque)
    ], dtype=np.float32)
    logit_opacities = np.array([-100.0, 100.0])  # front transparent, rear opaque

    splat = _make_splat(means, logit_opacities)
    cam = CameraPose(
        R=np.eye(3, dtype=np.float32),
        t=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        fx=50.0, fy=50.0,
        cx=32.0, cy=32.0,
        width=64, height=64,
    )
    projector = GaussianSplatProjector()
    result = projector.project([make_image_features()], [cam], splat)

    # Transparent front → no features; opaque rear → full features
    assert np.allclose(result[0], 0.0, atol=1e-6), (
        "Transparent front Gaussian should have zero features"
    )
    assert np.linalg.norm(result[1]) > 0.99, (
        "Rear opaque Gaussian behind transparent front should receive full features"
    )
