# API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## POST /ingest

Ingest a scene file and build a searchable feature index.

**Request**
```json
{
  "scene_path": "/data/scenes/garden.ply",
  "scene_type": "point_cloud",
  "camera_params": null,
  "config": {
    "run_undistortion": true,
    "run_depth": true,
    "aggregation": "mean",
    "clip_model": "ViT-B-32",
    "use_sam": false,
    "grounding_dino_prompts": null
  }
}
```

**`config.grounding_dino_prompts` — Grounding DINO pre-labelling**

When set to a non-empty list of strings, Grounding DINO localises the listed
object classes in each source image before dense CLIP feature lifting runs.
SAM then refines each bounding box into a precise per-object mask, so only
tiles that overlap a detected region are updated. This significantly reduces
background contamination in cluttered scenes.

Multiple class names are joined with ` . ` internally (Grounding DINO caption
format), so you can list them separately:

```json
"grounding_dino_prompts": ["chair", "table", "sofa"]
```

`grounding_dino_prompts` is independent of `use_sam`: both can be enabled
together (SAM auto-segments first, then DINO-prompted masks narrow regions
further) or used individually.

Requires `groundingdino` and SAM model weights. See `scripts/download_models.sh`.

**Response 200**
```json
{
  "scene_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "primitive_count": 95000,
  "feature_dim": 512,
  "scene_type": "point_cloud",
  "status": "ok"
}
```

---

## POST /query

Query a scene with natural language.

**Request**
```json
{
  "scene_id": "3fa85f64-...",
  "query": "red chair",
  "top_k": 100,
  "threshold": 0.25,
  "rerank": false
}
```

**Response 200**
```json
{
  "scene_id": "3fa85f64-...",
  "query": "red chair",
  "matches": [
    {"primitive_id": 42, "score": 0.91, "position_3d": [1.2, 0.5, -0.3]},
    {"primitive_id": 43, "score": 0.89, "position_3d": [1.3, 0.5, -0.3]}
  ],
  "query_time_ms": 12.4,
  "viewer_updated": true
}
```

---

## GET /scene/{scene_id}

Get metadata for a registered scene.

---

## GET /scene/{scene_id}/features

Get feature store statistics (primitive count, feature dim, index status).

---

## DELETE /scene/{scene_id}

Remove a scene and its feature index from disk.

---

## POST /agent/chat

Send a natural language message to the scene-query agent. The agent autonomously
decides which tools to call (query, count, highlight, measure, list) before
returning a plain-text reply. Pass `session_id` from the previous response to
continue a conversation.

**Request**
```json
{
  "message": "Count all chairs in scene room1 and highlight the ones near the window",
  "session_id": null
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `message` | string | yes | 1–2048 characters |
| `session_id` | string | no | Omit to start a new session |

**Response 200**
```json
{
  "reply": "Found 12 chairs total. 4 are within 0.8m of the window region and have been highlighted in the viewer.",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

**Multi-turn example**
```bash
# Turn 1 — start a session
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "List available scenes"}' | jq -r '.session_id')

# Turn 2 — continue the same conversation
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Query scene room1 for sofas\", \"session_id\": \"$SESSION\"}"
```

**Agent tools** — called automatically by the LLM:

| Tool | Description |
|------|-------------|
| `query_scene` | CLIP → FAISS search, returns matches + 3D positions |
| `count_matches` | Same search, returns count only |
| `highlight_primitives` | Sends IDs to threecrate viewer via IPC |
| `clear_highlights` | Clears all viewer highlights |
| `measure_distance` | Euclidean distance between two primitives |
| `list_scenes` | Lists scene IDs with saved indexes |

**Configuration** (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `SQ_AGENT_MODEL` | `qwen2.5:7b` | Ollama model name |
| `SQ_AGENT_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `SQ_AGENT_MAX_ITERATIONS` | `10` | Max tool-call steps per message |
| `SQ_AGENT_TEMPERATURE` | `0.0` | LLM sampling temperature |

**Errors**

| HTTP | Cause |
|------|-------|
| 500 | Ollama not running, model not pulled, or agent loop error |

Sessions are evicted after 1 hour of inactivity.

---

## GET /health

Health check.

**Response**
```json
{
  "status": "ok",
  "models_loaded": ["clip"],
  "viewer_connected": false
}
```

---

## Error codes

| HTTP | Error key | Meaning |
|------|-----------|---------|
| 404 | scene_not_found | scene_id not registered |
| 422 | ingestion_failed | Scene file invalid or unsupported |
| 422 | validation_error | Request body failed schema validation |
| 500 | query_failed | FAISS search or encoding error |
| 500 | internal_error | Unexpected server error |
