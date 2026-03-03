# Architecture

## Module Interaction

```
IngestRequest
    │
    ▼
SceneValidator ──► ValidationError
    │
    ▼
SceneLoaderFactory (PLYLoader / SplatLoader)
    │
    ▼
Scene (PointCloud | GaussianSplat)
    │
    ▼
PreprocessingPipeline
  ├── UndistortionStep
  └── DepthEstimationStep
    │
    ▼
FeatureLiftingPipeline
  ├── CLIPExtractor (tile → encode)
  ├── [SAMLifter]         ← primary path
  └── [PointCloudProjector] ← fallback
    │
    ▼
FeatureIndex.build()
    │
    ▼
IndexPersistence.save()
    │
    ▼
SceneRegistry (in-memory dict, Phase 1)

QueryRequest
    │
    ▼
CLIPTextEncoder.encode()
    │
    ▼
Searcher.search() → FAISS index
    │
    ▼
[SpatialReranker] (optional)
    │
    ▼
ViewerBridge.highlight() → Rust IPC → HighlightOverlay
```

## Key Design Decisions

### Python/Rust split
Python owns all ML computation (CLIP, SAM, FAISS). Rust owns rendering and 3D geometry. This avoids embedding a Python runtime in the render loop.

### FAISS tier selection
- `<100K` primitives → `IndexFlatIP` (exact, <50ms)
- `100K–1M` → `IndexIVFFlat` with 256 clusters
- `>1M` → `IndexHNSWFlat` (M=32)

All vectors are L2-normalized so inner product = cosine similarity.

### Graceful degradation
- SAM → direct projection fallback if SAM fails
- Viewer bridge → query still returns results if viewer is disconnected

### Phase 1 feature lifting
Phase 1 uses point cloud RGB colors projected through a fixed random matrix as a proxy for CLIP features. This lets the full ingestion/query pipeline run end-to-end without image data. Image-based CLIP lifting is wired in Phase 2.
