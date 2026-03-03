"""
Validates feature lifting quality using known scenes.
Run in CI to detect regressions.

Usage:
    uv run python scripts/validate_lifting.py --threshold 0.85
"""

from __future__ import annotations

import argparse
import sys


# Known (scene_name, query, expected_primitive_ids) tuples
# Populated in Phase 2 once real scenes are indexed
KNOWN_QUERIES: list[tuple[str, str, list[int]]] = [
    # ("garden_scene", "chair", [42, 43, 44, 45]),
    # ("kitchen_scene", "refrigerator", [112, 113]),
]


def compute_recall(matches: list, expected_ids: list[int]) -> float:
    matched_ids = {m["primitive_id"] for m in matches}
    hits = sum(1 for eid in expected_ids if eid in matched_ids)
    return hits / len(expected_ids) if expected_ids else 0.0


def query_scene(scene_name: str, query: str, top_k: int = 50) -> dict:
    import httpx

    resp = httpx.post(
        "http://localhost:8000/api/v1/query",
        json={"scene_id": scene_name, "query": query, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def validate(threshold: float = 0.85) -> None:
    if not KNOWN_QUERIES:
        print("No known queries configured — skipping validation (Phase 1)")
        return

    failures = []
    for scene_name, query, expected_ids in KNOWN_QUERIES:
        result = query_scene(scene_name, query)
        recall = compute_recall(result["matches"], expected_ids)
        status = "PASS" if recall >= threshold else "FAIL"
        print(f"[{status}] scene={scene_name!r} query={query!r} recall={recall:.2f} (threshold={threshold})")
        if recall < threshold:
            failures.append((scene_name, query, recall))

    if failures:
        print(f"\n{len(failures)} validation(s) failed")
        sys.exit(1)
    else:
        print("\nAll validations passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    validate(args.threshold)
