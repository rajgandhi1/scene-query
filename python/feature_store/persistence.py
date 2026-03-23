"""Save and load FAISS indexes to/from disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from python.feature_store.index import FeatureIndex
from python.utils.errors import FeatureStoreError
from python.utils.logging import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "feature_index.faiss"
POSITIONS_FILENAME = "positions.npy"


class IndexPersistence:
    """Handles saving and loading FeatureIndex objects."""

    def __init__(self, store_root: Path) -> None:
        self._root = store_root
        self._root.mkdir(parents=True, exist_ok=True)

    def _index_path(self, scene_id: str) -> Path:
        return self._root / scene_id / INDEX_FILENAME

    def _positions_path(self, scene_id: str) -> Path:
        return self._root / scene_id / POSITIONS_FILENAME

    def save(self, feature_index: FeatureIndex) -> Path:
        """
        Write the FAISS index and (if present) positions to disk.

        Args:
            feature_index: A built FeatureIndex.

        Returns:
            Path where the FAISS index was written.
        """
        try:
            import faiss

            if not feature_index.is_built:
                raise FeatureStoreError("Cannot save an unbuilt index")

            out_path = self._index_path(feature_index.scene_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(feature_index._index, str(out_path))
            logger.info("Index saved: %s (%d primitives)", out_path, feature_index.n_primitives)

            if feature_index.positions is not None:
                pos_path = self._positions_path(feature_index.scene_id)
                np.save(str(pos_path), feature_index.positions)
                logger.info("Positions saved: %s", pos_path)

            return out_path
        except Exception as exc:
            raise FeatureStoreError(f"Failed to save index: {exc}") from exc

    def load(self, scene_id: str) -> FeatureIndex:
        """
        Load a previously saved FAISS index and positions (if present).

        Args:
            scene_id: Scene identifier.

        Returns:
            FeatureIndex with the loaded FAISS index and positions.

        Raises:
            FeatureStoreError: If the index file does not exist.
        """
        try:
            import faiss

            index_path = self._index_path(scene_id)
            if not index_path.exists():
                raise FeatureStoreError(
                    f"No index found for scene '{scene_id}' at {index_path}"
                )
            raw = faiss.read_index(str(index_path))
            fi = FeatureIndex(scene_id)
            fi._index = raw
            fi._n_primitives = raw.ntotal
            fi._feature_dim = raw.d

            pos_path = self._positions_path(scene_id)
            if pos_path.exists():
                fi._positions = np.load(str(pos_path))
                logger.info("Positions loaded: %s (%d primitives)", pos_path, len(fi._positions))
            else:
                logger.warning("No positions file for scene '%s' — position_3d will be zero", scene_id)

            logger.info("Index loaded: %s (%d primitives)", index_path, raw.ntotal)
            return fi
        except FeatureStoreError:
            raise
        except Exception as exc:
            raise FeatureStoreError(f"Failed to load index for '{scene_id}': {exc}") from exc

    def exists(self, scene_id: str) -> bool:
        return self._index_path(scene_id).exists()

    def delete(self, scene_id: str) -> None:
        index_path = self._index_path(scene_id)
        if index_path.exists():
            index_path.unlink()
            logger.info("Deleted index: %s", index_path)
        pos_path = self._positions_path(scene_id)
        if pos_path.exists():
            pos_path.unlink()
            logger.info("Deleted positions: %s", pos_path)
