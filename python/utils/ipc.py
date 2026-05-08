"""Python ↔ Rust viewer IPC over Unix socket using MessagePack."""

import asyncio
import time
from pathlib import Path
from typing import Any, Literal

import msgpack

from python.utils.errors import IPCError
from python.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SOCKET_PATH = Path("/tmp/scene-query-viewer.sock")
CONNECT_TIMEOUT = 5.0
RESPONSE_TIMEOUT = 5.0
MAX_RECONNECT_ATTEMPTS = 5

_BACKOFF_INITIAL = 1.0
_BACKOFF_CAP = 30.0


class ViewerBridge:
    """
    Async IPC bridge to the Rust threecrate viewer.

    Sends highlight commands via MessagePack-encoded messages over a Unix
    domain socket. Gracefully degrades if the viewer is not connected.
    On disconnect, retries with exponential backoff (1s, 2s, 4s … capped at 30s)
    on each subsequent highlight call.

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
        self._status: Literal["connected", "disconnected", "reconnecting"] = "disconnected"
        self._backoff: float = _BACKOFF_INITIAL
        self._next_reconnect_at: float = 0.0

    @property
    def connected(self) -> bool:
        return self._status == "connected"

    @property
    def viewer_status(self) -> Literal["connected", "disconnected", "reconnecting"]:
        return self._status

    async def connect(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Attempt to connect to the viewer socket."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._socket_path)),
                timeout=timeout,
            )
            self._status = "connected"
            self._backoff = _BACKOFF_INITIAL
            logger.info("Connected to viewer at %s", self._socket_path)
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError, OSError) as exc:
            logger.warning("Viewer not available: %s — operating without live highlights", exc)
            self._status = "disconnected"

    def _on_disconnect(self) -> None:
        """Transition to reconnecting after an in-use connection loss."""
        self._status = "reconnecting"
        self._backoff = _BACKOFF_INITIAL
        self._next_reconnect_at = time.monotonic() + self._backoff
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        logger.warning("Viewer disconnected — will retry with exponential backoff")

    async def _try_reconnect(self) -> bool:
        """Attempt one reconnect if the backoff window has elapsed."""
        if time.monotonic() < self._next_reconnect_at:
            return False

        logger.debug(
            "Attempting reconnect to viewer at %s (backoff=%.1fs)",
            self._socket_path,
            self._backoff,
        )
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._socket_path)),
                timeout=CONNECT_TIMEOUT,
            )
            self._status = "connected"
            self._backoff = _BACKOFF_INITIAL
            logger.info("Reconnected to viewer at %s", self._socket_path)
            return True
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError, OSError) as exc:
            logger.debug("Reconnect attempt failed: %s", exc)
            self._backoff = min(self._backoff * 2, _BACKOFF_CAP)
            self._next_reconnect_at = time.monotonic() + self._backoff
            return False

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
        if self._status == "disconnected":
            return False

        if self._status == "reconnecting":
            if not await self._try_reconnect():
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
            self._on_disconnect()
            return False

    async def clear_highlights(self) -> bool:
        """Remove all current highlights from the viewer."""
        if self._status == "disconnected":
            return False

        if self._status == "reconnecting":
            if not await self._try_reconnect():
                return False

        try:
            await self._send({"op": "clear"})
            response = await self._recv()
            return response.get("status") == "ok"
        except IPCError as exc:
            logger.warning("IPC error during clear_highlights: %s", exc)
            self._on_disconnect()
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
        self._status = "disconnected"
        logger.info("Viewer bridge closed")
