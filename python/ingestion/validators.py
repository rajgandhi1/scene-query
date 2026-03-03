"""Scene input validation."""

from pathlib import Path

from python.utils.errors import ValidationError
from python.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".ply", ".obj", ".splat", ".las"}
MAX_FILE_SIZE_BYTES = 5 * 1024**3  # 5 GB


class SceneValidator:
    """Validates scene files before ingestion."""

    def validate(self, path: Path, scene_type: str) -> None:
        """
        Run all validation checks on the provided scene path.

        Args:
            path: Resolved, absolute path to the scene file or directory.
            scene_type: One of "point_cloud", "gaussian_splat", "mesh", "nerf".

        Raises:
            ValidationError: On any validation failure.
        """
        self._check_exists(path)
        self._check_extension(path, scene_type)
        self._check_size(path)
        logger.debug("Scene validated: %s", path)

    def _check_exists(self, path: Path) -> None:
        if not path.exists():
            raise ValidationError(f"Scene path does not exist: {path}")

    def _check_extension(self, path: Path, scene_type: str) -> None:
        if path.is_dir():
            # Image directories (e.g. COLMAP output) are allowed
            return
        if path.suffix not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )
        _EXPECTED = {
            "point_cloud": {".ply", ".las"},
            "gaussian_splat": {".splat", ".ply"},
            "mesh": {".obj"},
            "nerf": set(),  # directory-based
        }
        expected = _EXPECTED.get(scene_type, set())
        if expected and path.suffix not in expected:
            raise ValidationError(
                f"Extension '{path.suffix}' does not match scene_type '{scene_type}'. "
                f"Expected one of: {expected}"
            )

    def _check_size(self, path: Path) -> None:
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE_BYTES:
            size_gb = path.stat().st_size / 1024**3
            raise ValidationError(
                f"Scene file too large: {size_gb:.1f} GB (max 5 GB)"
            )
