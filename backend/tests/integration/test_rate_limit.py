"""The costly endpoints must be throttled; the cheap ones must not be."""

from __future__ import annotations

import pytest

from tests.integration.conftest import Harness


@pytest.mark.rate_limit(2)
async def test_the_chat_endpoint_is_throttled(harness: Harness):
    payload = {"question": "une question", "k": 1}

    statuses = [
        (await harness.client.post("/api/chat", json=payload)).status_code
        for _ in range(4)
    ]

    assert statuses[:2] == [200, 200]
    assert statuses[2:] == [429, 429]


@pytest.mark.rate_limit(2)
async def test_a_throttled_response_says_when_to_retry(harness: Harness):
    payload = {"question": "une question", "k": 1}
    for _ in range(2):
        await harness.client.post("/api/chat", json=payload)

    response = await harness.client.post("/api/chat", json=payload)

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    assert "Trop de requêtes" in response.json()["detail"]


@pytest.mark.rate_limit(1)
async def test_reads_are_not_throttled(harness: Harness):
    """Listing documents costs a query; throttling it would only degrade the UI."""
    await harness.client.post("/api/chat", json={"question": "q", "k": 1})

    statuses = [
        (await harness.client.get("/api/documents")).status_code for _ in range(5)
    ]

    assert statuses == [200] * 5


@pytest.mark.rate_limit(1)
async def test_ingestion_is_throttled_too(harness: Harness):
    """An upload runs an embedding model over the whole file."""
    files = {"file": ("a.txt", b"du contenu indexable", "text/plain")}

    first = await harness.client.post("/api/documents/upload", files=files)
    second = await harness.client.post("/api/documents/upload", files=files)

    assert first.status_code == 202
    assert second.status_code == 429


@pytest.mark.rate_limit(1)
async def test_health_probes_are_never_throttled(harness: Harness):
    """Throttling the readiness probe would take the service out of rotation."""
    statuses = [(await harness.client.get("/api/ready")).status_code for _ in range(5)]

    assert statuses == [200] * 5


@pytest.mark.rate_limit(1)
async def test_clients_are_throttled_independently(harness: Harness):
    """One noisy caller must not lock everyone else out."""
    payload = {"question": "une question", "k": 1}
    headers_a = {"X-Forwarded-For": "1.2.3.4"}
    headers_b = {"X-Forwarded-For": "5.6.7.8"}

    first = await harness.client.post("/api/chat", json=payload, headers=headers_a)
    repeat = await harness.client.post("/api/chat", json=payload, headers=headers_a)
    other = await harness.client.post("/api/chat", json=payload, headers=headers_b)

    assert (first.status_code, repeat.status_code, other.status_code) == (200, 429, 200)
