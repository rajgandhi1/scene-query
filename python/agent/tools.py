"""Tool definitions and executors for the scene-query agent."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from python.feature_store.persistence import IndexPersistence
from python.query_engine.encoder import CLIPTextEncoder
from python.query_engine.reranker import SpatialReranker
from python.query_engine.searcher import Searcher
from python.utils.ipc import ViewerBridge
from python.utils.logging import get_logger

logger = get_logger(__name__)

# OpenAI-compatible function tool definitions passed to the LLM
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_scene",
            "description": (
                "Search a 3D scene for primitives matching a text description. "
                "Returns primitive IDs, similarity scores, and 3D positions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {
                        "type": "string",
                        "description": "The scene to search.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what to find.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default 100).",
                        "default": 100,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum cosine similarity 0.0–1.0 (default 0.25).",
                        "default": 0.25,
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Apply spatial reranking to cluster nearby matches (default false).",
                        "default": False,
                    },
                },
                "required": ["scene_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_matches",
            "description": "Count how many primitives in a scene match a text description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "query": {"type": "string"},
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold (default 0.25).",
                        "default": 0.25,
                    },
                },
                "required": ["scene_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "highlight_primitives",
            "description": "Highlight specific primitives in the 3D viewer by their IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "primitive_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Primitive IDs to highlight.",
                    },
                    "scores": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Similarity scores for color mapping, same length as primitive_ids.",
                    },
                    "color_map": {
                        "type": "string",
                        "description": "Matplotlib colormap name (default 'plasma').",
                        "default": "plasma",
                    },
                },
                "required": ["primitive_ids", "scores"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_highlights",
            "description": "Remove all highlights from the 3D viewer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_distance",
            "description": "Measure Euclidean distance in 3D space between two primitives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "primitive_id_a": {"type": "integer"},
                    "primitive_id_b": {"type": "integer"},
                },
                "required": ["scene_id", "primitive_id_a", "primitive_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scenes",
            "description": "List all available scene IDs that can be queried.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ToolExecutor:
    """
    Executes agent tool calls using the scene-query backend.

    Wraps Searcher, ViewerBridge, and IndexPersistence so the agent loop
    stays free of direct ML/IPC imports.
    """

    def __init__(
        self,
        persistence: IndexPersistence,
        viewer_bridge: ViewerBridge | None = None,
    ) -> None:
        self._persistence = persistence
        self._viewer_bridge = viewer_bridge
        self._searcher = Searcher(persistence)
        self._encoder = CLIPTextEncoder()
        self._reranker = SpatialReranker()

    async def execute(self, tool_name: str, args: dict[str, Any]) -> Any:
        match tool_name:
            case "query_scene":
                return await self._query_scene(**args)
            case "count_matches":
                return await self._count_matches(**args)
            case "highlight_primitives":
                return await self._highlight_primitives(**args)
            case "clear_highlights":
                return await self._clear_highlights()
            case "measure_distance":
                return await self._measure_distance(**args)
            case "list_scenes":
                return self._list_scenes()
            case _:
                return {"error": f"Unknown tool: {tool_name}"}

    # --- Tool implementations ---

    async def _query_scene(
        self,
        scene_id: str,
        query: str,
        top_k: int = 100,
        threshold: float = 0.25,
        rerank: bool = False,
    ) -> dict[str, Any]:
        # Offload CPU-bound ops to thread pool so the event loop stays responsive
        embedding = await asyncio.to_thread(self._encoder.encode, query)
        results = await asyncio.to_thread(
            self._searcher.search, scene_id, embedding, top_k, threshold
        )
        if rerank and results:
            positions = self._searcher.get_positions(scene_id)
            if positions is not None:
                results = await asyncio.to_thread(
                    self._reranker.rerank, results, positions
                )

        # Cap returned matches to avoid flooding the context window
        _MATCH_CAP = 20
        return {
            "total_matches": len(results),
            "shown": min(len(results), _MATCH_CAP),
            "matches": [
                {
                    "primitive_id": r.primitive_id,
                    "score": round(r.score, 4),
                    "position_3d": [round(v, 4) for v in r.position_3d],
                }
                for r in results[:_MATCH_CAP]
            ],
        }

    async def _count_matches(
        self,
        scene_id: str,
        query: str,
        threshold: float = 0.25,
    ) -> dict[str, Any]:
        embedding = await asyncio.to_thread(self._encoder.encode, query)
        results = await asyncio.to_thread(
            self._searcher.search, scene_id, embedding, 10_000, threshold
        )
        return {"count": len(results), "query": query, "scene_id": scene_id}

    async def _highlight_primitives(
        self,
        primitive_ids: list[int],
        scores: list[float],
        color_map: str = "plasma",
    ) -> dict[str, Any]:
        if not self._viewer_bridge or not self._viewer_bridge.connected:
            return {"ok": False, "reason": "viewer not connected"}
        ok = await self._viewer_bridge.highlight(primitive_ids, scores, color_map)
        return {"ok": ok}

    async def _clear_highlights(self) -> dict[str, Any]:
        if not self._viewer_bridge or not self._viewer_bridge.connected:
            return {"ok": False, "reason": "viewer not connected"}
        ok = await self._viewer_bridge.clear_highlights()
        return {"ok": ok}

    async def _measure_distance(
        self,
        scene_id: str,
        primitive_id_a: int,
        primitive_id_b: int,
    ) -> dict[str, Any]:
        positions = self._searcher.get_positions(scene_id)
        if positions is None:
            return {"error": "No position data available for this scene"}
        pa = positions[primitive_id_a]
        pb = positions[primitive_id_b]
        dist = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(pa, pb)))
        return {
            "distance": round(dist, 4),
            "primitive_id_a": primitive_id_a,
            "primitive_id_b": primitive_id_b,
        }

    def _list_scenes(self) -> dict[str, Any]:
        scene_ids = self._persistence.list_scenes()
        return {"scenes": scene_ids, "count": len(scene_ids)}
