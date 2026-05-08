"""Unit tests for SceneRegistryDB — SQLite-backed persistence."""

from __future__ import annotations

import pytest

from python.db.registry import SceneRegistryDB

_META = {
    "scene_id": "scene-abc",
    "scene_type": "point_cloud",
    "primitive_count": 1000,
    "feature_dim": 512,
    "source_path": "/data/scenes/test.ply",
    "created_at": "2026-05-08T00:00:00+00:00",
}


@pytest.fixture
async def registry():
    reg = SceneRegistryDB("sqlite:///:memory:")
    await reg.initialize()
    yield reg
    await reg.close()


@pytest.mark.asyncio
async def test_set_and_get(registry):
    await registry.set("scene-abc", _META)
    result = await registry.get("scene-abc")
    assert result is not None
    assert result["scene_id"] == "scene-abc"
    assert result["primitive_count"] == 1000
    assert result["feature_dim"] == 512
    assert result["scene_type"] == "point_cloud"


@pytest.mark.asyncio
async def test_get_missing_returns_none(registry):
    assert await registry.get("nonexistent") is None


@pytest.mark.asyncio
async def test_contains_true_after_set(registry):
    assert not await registry.contains("scene-abc")
    await registry.set("scene-abc", _META)
    assert await registry.contains("scene-abc")


@pytest.mark.asyncio
async def test_delete_removes_entry(registry):
    await registry.set("scene-abc", _META)
    await registry.delete("scene-abc")
    assert not await registry.contains("scene-abc")
    assert await registry.get("scene-abc") is None


@pytest.mark.asyncio
async def test_set_overwrites_existing(registry):
    await registry.set("scene-abc", _META)
    updated = {**_META, "primitive_count": 2000}
    await registry.set("scene-abc", updated)
    result = await registry.get("scene-abc")
    assert result is not None
    assert result["primitive_count"] == 2000


@pytest.mark.asyncio
async def test_all_scenes_returns_all(registry):
    meta2 = {**_META, "scene_id": "scene-xyz", "created_at": "2026-05-08T01:00:00+00:00"}
    await registry.set("scene-abc", _META)
    await registry.set("scene-xyz", meta2)
    scenes = await registry.all_scenes()
    ids = {s["scene_id"] for s in scenes}
    assert ids == {"scene-abc", "scene-xyz"}


@pytest.mark.asyncio
async def test_all_scenes_empty(registry):
    assert await registry.all_scenes() == []


@pytest.mark.asyncio
async def test_persists_across_connections(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"

    reg1 = SceneRegistryDB(db_url)
    await reg1.initialize()
    await reg1.set("scene-abc", _META)
    await reg1.close()

    reg2 = SceneRegistryDB(db_url)
    await reg2.initialize()
    result = await reg2.get("scene-abc")
    await reg2.close()

    assert result is not None
    assert result["scene_id"] == "scene-abc"
    assert result["feature_dim"] == 512
