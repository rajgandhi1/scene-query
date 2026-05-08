#!/usr/bin/env python3
"""
Visual smoke-test for alpha-compositing aware Gaussian Splat feature weighting.

Creates three rows of Gaussians in a grid, each with a different opacity profile,
runs GaussianSplatProjector with a single camera, then plots which Gaussians
received features (green) vs were occluded/transparent (red).

Run with:
    uv run python scripts/visualize_alpha_weighting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from python.feature_lifting.clip_extractor import ImageFeatures
from python.feature_lifting.feature_projector import CameraPose, GaussianSplatProjector
from python.ingestion.loaders import GaussianSplat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_splat(means: np.ndarray, logit_opacities: np.ndarray) -> GaussianSplat:
    N = len(means)
    return GaussianSplat(
        means=means.astype(np.float32),
        scales=np.zeros((N, 3), dtype=np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
        opacities=logit_opacities.astype(np.float32),
        sh_coeffs=np.zeros((N, 1, 3), dtype=np.float32),
    )


def make_image_features(D: int = 64) -> ImageFeatures:
    """Tiny uniform feature grid — content doesn't matter for this test."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((8, 8, D)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb /= np.where(norms == 0, 1.0, norms)
    return ImageFeatures(embeddings=emb, image_hw=(448, 448), tile_size=112, tile_stride=56)


def make_camera() -> CameraPose:
    """Camera at the origin looking down +Z (R=I, t=0)."""
    return CameraPose(
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        fx=200.0, fy=200.0,
        cx=224.0, cy=224.0,
        width=448, height=448,
    )


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene():
    """
    Three rays of Gaussians, each along a constant x/z ratio so all Gaussians
    in the same ray project to the same image tile (transmittance is per-ray).

    Ray A  x/z = -0.4  "all opaque":    z=1..4, logit=+10
    Ray B  x/z =  0.0  "all transparent": z=1..4, logit=-10
    Ray C  x/z = +0.4  "opaque front, opaque rear":
                         z=1 opaque front → z=2,3,4 behind get T=0

    Camera at origin, R=I, looking down +Z.
    For u = fx*(x/z) + cx = 200*(x/z) + 224:
      ray A: u = 200*(-0.4) + 224 = 144  (tile_col 2)
      ray B: u = 200*(0.0)  + 224 = 224  (tile_col 4)
      ray C: u = 200*(+0.4) + 224 = 304  (tile_col 5)
    All in frame for z=1..4.
    """
    depth_levels = [1.0, 2.0, 3.0, 4.0]

    # Ray A: x/z = -0.4  → x = -0.4*z
    col_a_means = np.array([[-0.4 * z, 0.0, z] for z in depth_levels])
    col_a_logits = np.full(len(col_a_means), 10.0)

    # Ray B: x/z = 0  → x = 0
    col_b_means = np.array([[0.0, 0.0, z] for z in depth_levels])
    col_b_logits = np.full(len(col_b_means), -10.0)

    # Ray C: x/z = +0.4  → x = +0.4*z; z=1 is fully opaque (occludes z=2,3,4)
    col_c_means = np.array([[0.4 * z, 0.0, z] for z in depth_levels])
    col_c_logits = np.array([10.0, 10.0, 10.0, 10.0])

    all_means = np.concatenate([col_a_means, col_b_means, col_c_means])
    all_logits = np.concatenate([col_a_logits, col_b_logits, col_c_logits])
    labels = (
        [f"A z={z}" for z in depth_levels]
        + [f"B z={z}" for z in depth_levels]
        + [f"C z={z}" for z in depth_levels]
    )
    return make_splat(all_means, all_logits), all_means, all_logits, labels


# ---------------------------------------------------------------------------
# Run projector & plot
# ---------------------------------------------------------------------------

def main() -> None:
    splat, means, logit_opacities, labels = build_scene()
    alphas = 1.0 / (1.0 + np.exp(-logit_opacities))  # sigmoid

    projector = GaussianSplatProjector()
    result = projector.project([make_image_features()], [make_camera()], splat)

    norms = np.linalg.norm(result, axis=1)
    got_features = norms > 0

    # -----------------------------------------------------------------------
    # Plot 1: 3D scatter — colour by got_features
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))

    ax3d = fig.add_subplot(121, projection="3d")
    colors = ["#2ecc71" if f else "#e74c3c" for f in got_features]
    ax3d.scatter(means[:, 0], means[:, 2], means[:, 1],  # x, depth, y
                 c=colors, s=200, edgecolors="k", linewidths=0.5)
    for i, (pos, lbl) in enumerate(zip(means, labels)):
        ax3d.text(pos[0], pos[2] + 0.05, pos[1], lbl, fontsize=6)
    ax3d.set_xlabel("X (column)")
    ax3d.set_ylabel("Z (depth)")
    ax3d.set_zlabel("Y")
    ax3d.set_title("Gaussians: green = got features, red = zero features")
    green_patch = mpatches.Patch(color="#2ecc71", label="received features")
    red_patch = mpatches.Patch(color="#e74c3c", label="zero features (occluded / transparent)")
    ax3d.legend(handles=[green_patch, red_patch], loc="upper left", fontsize=8)

    # -----------------------------------------------------------------------
    # Plot 2: bar chart — alpha vs feature norm per Gaussian
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(122)
    x = np.arange(len(labels))
    width = 0.35

    bars_alpha = ax2.bar(x - width / 2, alphas, width, label="alpha = σ(logit)", color="#3498db", alpha=0.7)
    bars_norm = ax2.bar(x + width / 2, norms, width, label="feature L2 norm", color="#2ecc71", alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("value")
    ax2.set_title("Alpha vs feature assignment per Gaussian")
    ax2.legend()
    ax2.axhline(1e-4, color="gray", linestyle="--", linewidth=0.8, label="contrib threshold")

    fig.tight_layout()
    out = Path(__file__).parent / "alpha_weighting_visual.png"
    plt.savefig(out, dpi=130)
    print(f"Saved → {out}")

    # Summary to stdout
    print("\nGaussian  |  logit  |  alpha  |  norm  |  features?")
    print("-" * 56)
    for lbl, lo, al, nm, gf in zip(labels, logit_opacities, alphas, norms, got_features):
        print(f"{lbl:<10}  {lo:+6.1f}   {al:.4f}   {nm:.4f}   {'YES' if gf else 'NO'}")

    plt.show()


if __name__ == "__main__":
    main()
