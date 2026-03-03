"""Python ↔ Rust viewer IPC over Unix socket using MessagePack."""

import asyncio
import socket
from pathlib import Path
from typing import Any

import msgpack

from python.utils.errors import IPCError
from python.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SOCKET_PATH = Path("/tmp/scene-query-viewer.sock")
CONNECT_TIMEOUT = 5.0
RESPONSE_TIMEOUT = 5.0
MAX_RECONNECT_ATTEMPTS = 5


class ViewerBridge:
    """
    Async IPC bridge to the Rust threecrate viewer.

    Sends highlight commands via MessagePack-encoded messages over a Unix
    domain socket. Gracefully degrades if the viewer is not connected.

    Usage:
        bridge = ViewerBridge()
        await bridge.connect()
        await bridge.highlight(primitive_ids=[1, 2, 3], scores=[0.9, 0.8, 0.7])
        await bridge.close()
    """

    def __init__(self, socket_path: Path = DEFAULT_SOCKET_PATH) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Attempt to connect to the viewer socket."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._socket_path)),
                timeout=timeout,
            )
            self._connected = True
            logger.info("Connected to viewer at %s", self._socket_path)
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError) as exc:
            logger.warning("Viewer not available: %s — operating without live highlights", exc)
            self._connected = False

    async def highlight(
        self,
        primitive_ids: list[int],
        scores: list[float],
        color_map: str = "plasma",
    ) -> bool:
        """
        Send a highlight command to the viewer.

        Args:
            primitive_ids: 3D primitive IDs to highlight.
            scores: Similarity scores in [0, 1], same length as primitive_ids.
            color_map: Matplotlib-compatible colormap name.

        Returns:
            True if the viewer acknowledged, False if viewer is unavailable.
        """
        if not self._connected:
            return False

        message = {
            "op": "highlight",
            "primitive_ids": primitive_ids,
            "scores": scores,
            "color_map": color_map,
        }
        try:
            await self._send(message)
            response = await self._recv()
            return response.get("status") == "ok"
        except IPCError as exc:
            logger.warning("IPC error during highlight: %s", exc)
            self._connected = False
            return False

    async def clear_highlights(self) -> bool:
        """Remove all current highlights from the viewer."""
        if not self._connected:
            return False
        try:
            await self._send({"op": "clear"})
            response = await self._recv()
            return response.get("status") == "ok"
        except IPCError:
            self._connected = False
            return False

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._writer is None:
            raise IPCError("Not connected to viewer")
        try:
            data = msgpack.packb(payload, use_bin_type=True)
            # 4-byte length prefix + payload
            length = len(data).to_bytes(4, "big")
            self._writer.write(length + data)
            await self._writer.drain()
        except Exception as exc:
            raise IPCError(f"Send failed: {exc}") from exc

    async def _recv(self) -> dict[str, Any]:
        if self._reader is None:
            raise IPCError("Not connected to viewer")
        try:
            raw_len = await asyncio.wait_for(self._reader.readexactly(4), RESPONSE_TIMEOUT)
            length = int.from_bytes(raw_len, "big")
            data = await asyncio.wait_for(self._reader.readexactly(length), RESPONSE_TIMEOUT)
            result: dict[str, Any] = msgpack.unpackb(data, raw=False)
            return result
        except Exception as exc:
            raise IPCError(f"Recv failed: {exc}") from exc

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False
        logger.info("Viewer bridge closed")
