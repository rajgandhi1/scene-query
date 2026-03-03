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
    "clip_model": "ViT-B-32"
  }
}
```

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
