"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_point_cloud():
    """1 000-point synthetic point cloud."""
    from python.ingestion.loaders import PointCloud

    rng = np.random.default_rng(0)
    return PointCloud(
        points=rng.uniform(-1, 1, (1000, 3)).astype(np.float32),
        colors=rng.uniform(0, 1, (1000, 3)).astype(np.float32),
    )


@pytest.fixture
def tiny_gaussian_splat():
    """100-Gaussian synthetic splat."""
    from python.ingestion.loaders import GaussianSplat

    rng = np.random.default_rng(1)
    N = 100
    return GaussianSplat(
        means=rng.uniform(-1, 1, (N, 3)).astype(np.float32),
        scales=rng.uniform(-3, 0, (N, 3)).astype(np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
        opacities=rng.uniform(0, 1, N).astype(np.float32),
        sh_coeffs=rng.uniform(-1, 1, (N, 1, 3)).astype(np.float32),
    )


@pytest.fixture
def random_features():
    """(200, 512) float32 L2-normalized feature matrix."""
    rng = np.random.default_rng(2)
    feat = rng.standard_normal((200, 512)).astype(np.float32)
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / norms


@pytest.fixture
def camera_params_json() -> dict:
    import json
    return json.loads((FIXTURES_DIR / "camera_params.json").read_text())


@pytest.fixture
def api_client():
    """FastAPI TestClient with app fully wired."""
    from python.api.app import create_app
    from python.models.registry import ModelRegistry

    ModelRegistry.reset()
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
