# scene-query

> Language-Queryable 3D Scenes — Natural language queries over Gaussian Splats and Point Clouds

Query a 3D scene with plain text and get spatially highlighted results in an interactive viewer.

```
"where is the red chair?" → highlighted region in 3D
```

---

## Quick Start

```bash
# Docker (recommended)
docker compose -f docker/docker-compose.yml up

# Or run locally
uv sync --extra dev
uv run uvicorn python.api.app:app --reload
```

## API Usage

```bash
# Ingest a scene
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"scene_path": "/data/garden.ply", "scene_type": "point_cloud"}'

# Query it
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "<id>", "query": "red chair", "top_k": 100}'
```

## Supported Formats

| Format | Type | Status |
|--------|------|--------|
| `.ply` | Point Cloud | Phase 1 |
| `.splat` | Gaussian Splat | Phase 3 |
| `.obj` | Mesh | Phase 3 |
| NeRF | NeRF | Future |

## Architecture

See [docs/architecture.md](docs/architecture.md) for full design details.

```
Images / Point Cloud / Gaussian Splat
            ↓
    Feature Lifting (CLIP → 3D)
            ↓
    FAISS Feature Store
            ↓
  Text Query → CLIP embedding → Similarity Search
            ↓
  Highlighted output in threecrate viewer
```

## Tech Stack

- **ML Pipeline**: Python 3.11+, CLIP (OpenCLIP), SAM, Grounding DINO
- **3D Viewer**: Rust (threecrate)
- **Feature Store**: FAISS
- **API**: FastAPI
- **IPC**: MessagePack over Unix socket

## Contributing

1. Fork and clone
2. `uv sync --extra dev`
3. `pre-commit install`
4. Cut a branch from `dev`, open a PR back to `dev`

## License

MIT
