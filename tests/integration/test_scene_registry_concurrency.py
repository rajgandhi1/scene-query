"""Stress test: 10 concurrent writes to SceneRegistry must not corrupt state."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from python.api.routes.ingest import SceneRegistry


@pytest.mark.asyncio
async def test_concurrent_set_no_corruption() -> None:
    """10 concurrent set() calls must all land without KeyError or missing data."""
    registry = SceneRegistry()
    n = 10

    async def _write(i: int) -> str:
        scene_id = f"scene-{i:04d}"
        await registry.set(scene_id, {
            "scene_id": scene_id,
            "scene_type": "point_cloud",
            "primitive_count": i * 100,
            "feature_dim": 512,
            "source_path": f"/tmp/scene_{i}.ply",
            "created_at": datetime.now(UTC).isoformat(),
        })
        return scene_id

    scene_ids = await asyncio.gather(*[_write(i) for i in range(n)])

    # Every write must be readable without KeyError or partial state
    for sid in scene_ids:
        meta = await registry.get(sid)
        assert meta is not None, f"{sid} missing from registry"
        assert meta["scene_id"] == sid
        assert "primitive_count" in meta
        assert "feature_dim" in meta


@pytest.mark.asyncio
async def test_concurrent_set_and_get_no_keyerror() -> None:
    """Interleaved reads and writes must not raise KeyError or return stale None."""
    registry = SceneRegistry()

    async def _write(scene_id: str) -> None:
        await registry.set(scene_id, {
            "scene_id": scene_id,
            "scene_type": "gaussian_splat",
            "primitive_count": 500,
            "feature_dim": 512,
            "source_path": "/tmp/dummy.splat",
            "created_at": datetime.now(UTC).isoformat(),
        })

    async def _read_after_write(scene_id: str) -> None:
        await _write(scene_id)
        meta = await registry.get(scene_id)
        # After awaiting our own write, the entry must exist
        assert meta is not None, f"{scene_id} vanished after write"

    await asyncio.gather(*[_read_after_write(f"s-{i}") for i in range(10)])


@pytest.mark.asyncio
async def test_concurrent_delete_no_double_delete() -> None:
    """Only one of two concurrent deletes should succeed; the other must see False from contains()."""
    registry = SceneRegistry()
    await registry.set("shared", {"scene_id": "shared", "scene_type": "point_cloud",
                                   "primitive_count": 1, "feature_dim": 512,
                                   "source_path": "/tmp/x.ply",
                                   "created_at": datetime.now(UTC).isoformat()})

    errors: list[Exception] = []

    async def _try_delete() -> None:
        if await registry.contains("shared"):
            try:
                await registry.delete("shared")
            except KeyError:
                errors.append(KeyError("shared"))

    await asyncio.gather(_try_delete(), _try_delete())
    # At most one KeyError is acceptable (TOCTOU between contains+delete),
    # but with the lock held through each atomic op there should be zero.
    assert len(errors) == 0, f"Unexpected KeyError(s) during concurrent delete: {errors}"
