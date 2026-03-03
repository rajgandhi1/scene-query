"""Custom exception hierarchy for scene-query."""


class SceneQueryError(Exception):
    """Base exception for all scene-query errors."""


class IngestionError(SceneQueryError):
    """Raised when scene loading or preprocessing fails."""


class ValidationError(IngestionError):
    """Raised when scene input fails validation."""


class FeatureLiftingError(SceneQueryError):
    """Raised when CLIP projection or feature assignment fails."""


class QueryError(SceneQueryError):
    """Raised when query encoding or similarity search fails."""


class ModelLoadError(SceneQueryError):
    """Raised when a model cannot be loaded from disk or registry."""


class IPCError(SceneQueryError):
    """Raised when Python↔Rust viewer communication fails."""


class FeatureStoreError(SceneQueryError):
    """Raised when FAISS index operations fail."""


class SceneNotFoundError(SceneQueryError):
    """Raised when a requested scene_id does not exist in the registry."""
