"""`read_capped` guards the URL import against oversized remote pages."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.errors import PayloadTooLargeError
from app.ingestion import read_capped


async def _stream(*pieces: bytes) -> AsyncIterator[bytes]:
    for piece in pieces:
        yield piece


async def test_returns_the_whole_body_when_under_the_limit():
    assert await read_capped(_stream(b"abc", b"def"), limit=100, what="Page") == b"abcdef"


async def test_empty_stream_returns_empty_bytes():
    assert await read_capped(_stream(), limit=10, what="Page") == b""


async def test_raises_once_the_limit_is_exceeded():
    with pytest.raises(PayloadTooLargeError, match="volumineux"):
        await read_capped(_stream(b"x" * 6, b"x" * 6), limit=10, what="Page")


async def test_stops_reading_instead_of_draining_the_stream():
    """The point is to stop early: a caller must not pay for the whole body."""
    consumed = 0

    async def counting() -> AsyncIterator[bytes]:
        nonlocal consumed
        for _ in range(100):
            consumed += 1
            yield b"y" * 10

    with pytest.raises(PayloadTooLargeError):
        await read_capped(counting(), limit=25, what="Page")

    # 3 chunks of 10 bytes is the first total above 25; nothing beyond is read.
    assert consumed == 3
