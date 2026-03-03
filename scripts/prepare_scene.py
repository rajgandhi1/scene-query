"""
Prepare and ingest a scene file via the API.

Usage:
    uv run python scripts/prepare_scene.py --path /data/garden.ply --type point_cloud
"""

from __future__ import annotations

import argparse
import json
import sys


def prepare(scene_path: str, scene_type: str, api_url: str) -> None:
    import httpx

    print(f"Ingesting: {scene_path} (type={scene_type})")
    resp = httpx.post(
        f"{api_url}/api/v1/ingest",
        json={"scene_path": scene_path, "scene_type": scene_type},
        timeout=3600,  # ingestion can be slow for large scenes
    )
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    print(json.dumps(data, indent=2))
    print(f"\nScene ingested successfully. scene_id: {data['scene_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to scene file")
    parser.add_argument(
        "--type",
        default="point_cloud",
        choices=["point_cloud", "gaussian_splat", "mesh", "nerf"],
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    prepare(args.path, args.type, args.api_url)
