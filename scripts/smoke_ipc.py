"""Manual smoke test for the Python→Rust IPC highlight bridge.

Usage:
    uv run python scripts/smoke_ipc.py

Requires the Rust IPC socket to be listening at /tmp/scene-query-viewer.sock.
Run the integration test suite first to confirm unit-level behavior:
    cargo test --test ipc_round_trip
"""

import asyncio

from python.utils.ipc import ViewerBridge


async def main() -> None:
    bridge = ViewerBridge()
    await bridge.connect()

    if not bridge.connected:
        print("SKIP: viewer socket not running — start the Rust viewer first")
        return

    print("Connected to viewer socket")

    ok = await bridge.highlight([0, 1, 2], [0.9, 0.7, 0.5], color_map="plasma")
    print(f"highlight (plasma)  ack: {ok}")

    ok = await bridge.highlight([3, 4], [1.0, 0.6], color_map="viridis")
    print(f"highlight (viridis) ack: {ok}")

    ok = await bridge.highlight([5], [0.8], color_map="hot")
    print(f"highlight (hot)     ack: {ok}")

    ok = await bridge.clear_highlights()
    print(f"clear               ack: {ok}")

    await bridge.close()
    print("Done")


if __name__ == "__main__":
    asyncio.run(main())
