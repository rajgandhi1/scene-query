"""Unit tests for ViewerBridge reconnection with exponential backoff."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from python.utils.ipc import _BACKOFF_CAP, _BACKOFF_INITIAL, ViewerBridge


def _make_bridge() -> ViewerBridge:
    return ViewerBridge()


# ---------------------------------------------------------------------------
# Initial connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_status_is_disconnected():
    bridge = _make_bridge()
    assert bridge.viewer_status == "disconnected"
    assert not bridge.connected


@pytest.mark.asyncio
async def test_connect_success_sets_connected():
    bridge = _make_bridge()
    mock_reader = MagicMock()
    mock_writer = MagicMock()
    mock_writer.close = MagicMock()

    with patch("asyncio.open_unix_connection", new=AsyncMock(return_value=(mock_reader, mock_writer))):
        await bridge.connect()

    assert bridge.viewer_status == "connected"
    assert bridge.connected


@pytest.mark.asyncio
async def test_connect_failure_stays_disconnected():
    bridge = _make_bridge()

    with patch("asyncio.open_unix_connection", side_effect=FileNotFoundError("no socket")):
        await bridge.connect()

    assert bridge.viewer_status == "disconnected"
    assert not bridge.connected


# ---------------------------------------------------------------------------
# Disconnect and backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ipc_error_transitions_to_reconnecting():
    bridge = _make_bridge()
    bridge._status = "connected"
    bridge._on_disconnect()

    assert bridge.viewer_status == "reconnecting"
    assert bridge._backoff == _BACKOFF_INITIAL


@pytest.mark.asyncio
async def test_highlight_returns_false_when_disconnected():
    bridge = _make_bridge()
    assert bridge.viewer_status == "disconnected"
    result = await bridge.highlight([1, 2], [0.9, 0.8])
    assert result is False


@pytest.mark.asyncio
async def test_highlight_returns_false_when_reconnecting_and_backoff_not_elapsed():
    bridge = _make_bridge()
    bridge._status = "reconnecting"
    bridge._next_reconnect_at = time.monotonic() + 100.0  # far future

    result = await bridge.highlight([1], [0.9])
    assert result is False


@pytest.mark.asyncio
async def test_reconnect_success_on_highlight_call():
    bridge = _make_bridge()
    bridge._status = "reconnecting"
    bridge._next_reconnect_at = 0.0  # backoff elapsed immediately

    mock_reader = MagicMock()
    mock_writer = AsyncMock()
    mock_writer.write = MagicMock()

    packed_ok = __import__("msgpack").packb({"status": "ok"}, use_bin_type=True)
    length_prefix = len(packed_ok).to_bytes(4, "big")
    mock_reader.readexactly = AsyncMock(side_effect=[length_prefix, packed_ok])

    with patch("asyncio.open_unix_connection", new=AsyncMock(return_value=(mock_reader, mock_writer))):
        result = await bridge.highlight([1], [0.9])

    assert bridge.viewer_status == "connected"
    assert result is True


@pytest.mark.asyncio
async def test_reconnect_failure_doubles_backoff():
    bridge = _make_bridge()
    bridge._status = "reconnecting"
    bridge._backoff = _BACKOFF_INITIAL
    bridge._next_reconnect_at = 0.0

    with patch("asyncio.open_unix_connection", side_effect=ConnectionRefusedError("refused")):
        result = await bridge._try_reconnect()

    assert result is False
    assert bridge._backoff == _BACKOFF_INITIAL * 2
    assert bridge.viewer_status == "reconnecting"


@pytest.mark.asyncio
async def test_backoff_caps_at_max():
    bridge = _make_bridge()
    bridge._status = "reconnecting"
    bridge._backoff = _BACKOFF_CAP
    bridge._next_reconnect_at = 0.0

    with patch("asyncio.open_unix_connection", side_effect=ConnectionRefusedError("refused")):
        await bridge._try_reconnect()

    assert bridge._backoff == _BACKOFF_CAP


@pytest.mark.asyncio
async def test_backoff_resets_after_successful_reconnect():
    bridge = _make_bridge()
    bridge._status = "reconnecting"
    bridge._backoff = 16.0
    bridge._next_reconnect_at = 0.0

    mock_reader, mock_writer = MagicMock(), AsyncMock()
    with patch("asyncio.open_unix_connection", new=AsyncMock(return_value=(mock_reader, mock_writer))):
        result = await bridge._try_reconnect()

    assert result is True
    assert bridge._backoff == _BACKOFF_INITIAL
    assert bridge.viewer_status == "connected"


# ---------------------------------------------------------------------------
# Highlight triggers reconnecting state on IPC error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_highlight_ipc_error_transitions_to_reconnecting():
    bridge = _make_bridge()
    bridge._status = "connected"
    bridge._reader = MagicMock()
    bridge._writer = MagicMock()
    bridge._writer.write = MagicMock()
    bridge._writer.drain = AsyncMock(side_effect=OSError("broken pipe"))

    result = await bridge.highlight([1], [0.9])

    assert result is False
    assert bridge.viewer_status == "reconnecting"
