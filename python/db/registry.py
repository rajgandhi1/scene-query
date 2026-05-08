"""SQLite-backed persistent scene registry."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from python.utils.logging import get_logger

logger = get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scenes (
    scene_id        TEXT PRIMARY KEY,
    scene_type      TEXT NOT NULL,
    primitive_count INTEGER NOT NULL,
    feature_dim     INTEGER NOT NULL,
    source_path     TEXT NOT NULL,
    created_at      TEXT NOT NULL
)
"""


def _db_path_from_url(db_url: str) -> str:
    """Return the filesystem path (or ':memory:') from a sqlite:/// URL."""
    if ":///:memory:" in db_url:
        return ":memory:"
    if db_url.startswith("sqlite"):
        return db_url.split("///", 1)[-1]
    raise ValueError(f"Unsupported db_url: {db_url!r} (only sqlite:/// is supported)")


class SceneRegistryDB:
    """
    Async scene registry backed by SQLite.

    Keeps a single persistent aiosqlite connection so that :memory: databases
    work correctly in tests. Call initialize() before use and close() on shutdown.
    """

    def __init__(self, db_url: str) -> None:
        self._db_path = _db_path_from_url(db_url)
        self._lock = asyncio.Lock()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and create the schema if not present."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        await self._db.commit()
        logger.info("Scene registry DB ready: %s", self._db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SceneRegistryDB not initialized — call initialize() first")
        return self._db

    async def get(self, scene_id: str) -> dict[str, Any] | None:
        async with self._lock:
            async with self._conn().execute(
                "SELECT * FROM scenes WHERE scene_id = ?", (scene_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row is not None else None

    async def set(self, scene_id: str, meta: dict[str, Any]) -> None:
        async with self._lock:
            await self._conn().execute(
                """INSERT OR REPLACE INTO scenes
                   (scene_id, scene_type, primitive_count, feature_dim, source_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(meta["scene_id"]),
                    str(meta["scene_type"]),
                    int(meta["primitive_count"]),  # type: ignore[arg-type]
                    int(meta["feature_dim"]),  # type: ignore[arg-type]
                    str(meta["source_path"]),
                    str(meta["created_at"]),
                ),
            )
            await self._conn().commit()

    async def delete(self, scene_id: str) -> None:
        async with self._lock:
            await self._conn().execute(
                "DELETE FROM scenes WHERE scene_id = ?", (scene_id,)
            )
            await self._conn().commit()

    async def contains(self, scene_id: str) -> bool:
        async with self._lock:
            async with self._conn().execute(
                "SELECT 1 FROM scenes WHERE scene_id = ?", (scene_id,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def all_scenes(self) -> list[dict[str, Any]]:
        """Return metadata for all registered scenes, ordered by creation time."""
        async with self._lock:
            async with self._conn().execute(
                "SELECT * FROM scenes ORDER BY created_at"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
