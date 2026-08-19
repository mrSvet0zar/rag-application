"""Oversized request bodies must be rejected, not absorbed.

The endpoint receives an already-parsed `UploadFile`, so any check written
inside it would run *after* the whole upload had been buffered. These tests
therefore pin the behaviour of the ASGI-level guard, including the case where
the client declares no Content-Length at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.integration.conftest import Harness

LIMIT = 2048


@pytest.mark.max_body(LIMIT)
async def test_upload_over_the_limit_is_rejected(harness: Harness):
    oversized = b"x" * (LIMIT * 2)

    response = await harness.client.post(
        "/api/documents/upload",
        files={"file": ("big.txt", oversized, "text/plain")},
    )

    assert response.status_code == 413
    assert "volumineuse" in response.json()["detail"].lower()


@pytest.mark.max_body(LIMIT)
async def test_upload_under_the_limit_still_works(harness: Harness):
    """The guard must not break ordinary uploads."""
    response = await harness.client.post(
        "/api/documents/upload",
        files={"file": ("small.txt", b"Un contenu court mais reel.", "text/plain")},
    )

    assert response.status_code == 202, response.text
    document = (
        await harness.client.get(f"/api/documents/{response.json()['id']}")
    ).json()
    assert document["total_chunks"] >= 1


@pytest.mark.max_body(LIMIT)
async def test_streamed_body_without_content_length_is_rejected(harness: Harness):
    """A chunked upload declares no length, so it can only be caught mid-stream.

    httpx encodes the multipart body, which is then replayed as an async
    iterator: that makes httpx use chunked transfer encoding and omit
    Content-Length — exactly the case the declared-length check cannot see.
    """
    encoded = httpx.Request(
        "POST",
        "http://test/api/documents/upload",
        files={"file": ("big.txt", b"y" * (LIMIT * 4), "text/plain")},
    )
    encoded.read()  # materialise the streaming multipart body
    body = encoded.content

    async def chunks() -> AsyncIterator[bytes]:
        for start in range(0, len(body), LIMIT):
            yield body[start : start + LIMIT]

    response = await harness.client.post(
        "/api/documents/upload",
        content=chunks(),
        headers={"Content-Type": encoded.headers["content-type"]},
    )

    assert response.status_code == 413


@pytest.mark.max_body(LIMIT)
async def test_nothing_is_persisted_when_an_upload_is_rejected(harness: Harness):
    await harness.client.post(
        "/api/documents/upload",
        files={"file": ("big.txt", b"z" * (LIMIT * 2), "text/plain")},
    )

    listing = await harness.client.get("/api/documents")
    assert listing.json() == []


@pytest.mark.max_body(LIMIT)
async def test_oversized_remote_page_is_rejected(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
):
    """The URL import must stop downloading, not check the size afterwards."""
    from tests.integration.test_api import _patch_fetch

    _patch_fetch(monkeypatch, f"<html><body><p>{'contenu ' * LIMIT}</p></body></html>")

    response = await harness.client.post(
        "/api/documents/import-url", json={"url": "https://example.com/huge"}
    )

    assert response.status_code == 413
    assert (await harness.client.get("/api/documents")).json() == []
