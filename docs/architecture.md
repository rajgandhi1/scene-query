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

AgentChatRequest
    │
    ▼
AgentSession (message history, context window trim)
    │
    ▼
AgentLoop → Qwen via Ollama (tool_choice="auto")
    │  ↕ tool calls
    ├── query_scene          → CLIPTextEncoder + Searcher (asyncio.to_thread)
    ├── count_matches        → CLIPTextEncoder + Searcher (asyncio.to_thread)
    ├── highlight_primitives → ViewerBridge
    ├── clear_highlights     → ViewerBridge
    ├── measure_distance     → positions array
    └── list_scenes          → IndexPersistence
    │
    ▼
Natural language reply + viewer updated
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

### Agent loop (12-factor design)
The agent module follows principles from [12-factor agents](https://github.com/humanlayer/12-factor-agents):

| Factor | Implementation |
|--------|---------------|
| Own your context window (3) | `AgentSession.trim()` caps history at 40 messages before every LLM call |
| Tools are structured outputs (4) | Tool definitions are plain JSON schema dicts; Qwen outputs validated tool calls |
| Unify execution and state (5) | `ToolExecutor` is a lifespan singleton — `Searcher`'s index cache persists across all requests |
| Own your control flow (8) | Explicit `for step in range(max_iterations)` loop; no framework magic |
| Compact errors into context (9) | Tool errors serialised as `[tool_error] name: ExcType: msg`, not raw tracebacks |

CPU-bound tool operations (CLIP encode, FAISS search) are offloaded via `asyncio.to_thread` so the FastAPI event loop stays responsive during agent steps.
