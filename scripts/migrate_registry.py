#!/usr/bin/env python3
"""Migrate existing on-disk FAISS indexes into the scene registry DB.

Run after upgrading to a version that introduces DB-backed scene persistence.
For each on-disk index that is absent from the DB, a row is inserted with
recoverable fields (primitive_count, feature_dim) and sentinel values for
fields that cannot be recovered (scene_type='unknown', source_path='unknown').
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.api.schemas import settings
from python.db.registry import SceneRegistryDB
from python.feature_store.persistence import IndexPersistence
from python.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def migrate() -> None:
    configure_logging(level=settings.log_level)

    persistence = IndexPersistence(store_root=settings.index_root)
    registry = SceneRegistryDB(db_url=settings.db_url)
    await registry.initialize()

    on_disk = persistence.list_scenes()
    logger.info("Found %d on-disk index(es) at %s", len(on_disk), settings.index_root)

    migrated = 0
    skipped = 0
    for scene_id in on_disk:
        if await registry.contains(scene_id):
            logger.debug("Scene %s already in registry — skipping", scene_id)
            skipped += 1
            continue

        fi = persistence.load(scene_id)
        await registry.set(scene_id, {
            "scene_id": scene_id,
            "scene_type": "unknown",
            "primitive_count": fi.n_primitives,
            "feature_dim": fi.feature_dim,
            "source_path": "unknown",
            "created_at": datetime.now(UTC).isoformat(),
        })
        logger.info(
            "Migrated scene %s: %d primitives, dim=%d",
            scene_id, fi.n_primitives, fi.feature_dim,
        )
        migrated += 1

    await registry.close()
    logger.info(
        "Migration complete: %d migrated, %d already present, %d total",
        migrated, skipped, len(on_disk),
    )


if __name__ == "__main__":
    asyncio.run(migrate())
