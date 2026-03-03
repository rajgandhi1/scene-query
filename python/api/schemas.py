"""Pydantic schemas for all API request and response types."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    scene_root: Path = Path("data/scenes")
    index_root: Path = Path("data/indexes")
    socket_path: Path = Path("/tmp/scene-query-viewer.sock")
    log_level: str = "INFO"
    max_file_size_gb: float = 5.0
    query_rate_limit: int = 60  # requests per minute per IP

    model_config = {"env_prefix": "SQ_"}


settings = Settings()


# --- Shared sub-models ---

class CameraParams(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: list[float] = Field(default_factory=list)


class IngestionConfig(BaseModel):
    run_undistortion: bool = True
    run_depth: bool = True
    aggregation: Literal["mean", "max"] = "mean"
    clip_model: str = "ViT-B-32"


class BoundingBox3D(BaseModel):
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float


# --- Request schemas ---

class IngestRequest(BaseModel):
    scene_path: Path
    scene_type: Literal["gaussian_splat", "point_cloud", "mesh", "nerf"]
    camera_params: CameraParams | None = None
    config: IngestionConfig = Field(default_factory=IngestionConfig)

    @field_validator("scene_path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        resolved = v.resolve()
        allowed_root = settings.scene_root.resolve()
        if not str(resolved).startswith(str(allowed_root)):
            raise ValueError(f"Path '{resolved}' is outside allowed root '{allowed_root}'")
        allowed_exts = {".ply", ".obj", ".splat", ".las"}
        if resolved.is_file() and resolved.suffix not in allowed_exts:
            raise ValueError(f"Unsupported extension '{resolved.suffix}'")
        return resolved


class QueryRequest(BaseModel):
    scene_id: str
    query: str = Field(min_length=1, max_length=512)
    top_k: int = Field(default=100, ge=1, le=10_000)
    threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    rerank: bool = False


# --- Response schemas ---

class IngestResponse(BaseModel):
    scene_id: str
    primitive_count: int
    feature_dim: int
    scene_type: str
    status: Literal["ok", "error"]
    message: str = ""


class QueryMatch(BaseModel):
    primitive_id: int
    score: float
    position_3d: tuple[float, float, float]
    bbox_3d: BoundingBox3D | None = None


class QueryResponse(BaseModel):
    scene_id: str
    query: str
    matches: list[QueryMatch]
    query_time_ms: float
    viewer_updated: bool = False


class SceneMetadata(BaseModel):
    scene_id: str
    scene_type: str
    primitive_count: int
    feature_dim: int
    source_path: str
    created_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: list[str]
    viewer_connected: bool
