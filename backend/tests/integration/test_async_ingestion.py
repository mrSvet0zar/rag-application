"""Ingestion runs detached from the request, which creates failure modes a
synchronous version never had: a document can be observed mid-flight, and a
restart can strand one forever."""

from __future__ import annotations

import asyncio

import pytest

from app.chunking import TextChunker
from app.ingestor import DocumentIngestor
from tests.doubles import FakeEmbedder
from tests.integration.conftest import Harness


async def test_a_document_is_visible_while_it_is_still_processing(harness: Harness):
    """The row exists immediately, so the client has something to poll."""
    ingestor = harness.services.ingestor

    document = await ingestor.create_pending(
        filename="lent.md",
        content_type="text/plain",
        size_bytes=42,
        text="du contenu indexable",
    )

    assert document["status"] == "processing"
    listed = (await harness.client.get(f"/api/documents/{document['id']}")).json()
    assert listed["status"] == "processing"
    assert listed["total_chunks"] == 0


async def test_processing_moves_the_document_to_completed(harness: Harness):
    ingestor = harness.services.ingestor
    document = await ingestor.create_pending(
        filename="a.md", content_type="text/plain", size_bytes=10, text="du contenu"
    )

    await ingestor.process(document["id"], "a.md", "du contenu")

    updated = await harness.db.get_document(document["id"])
    assert updated["status"] == "completed"
    assert updated["total_chunks"] >= 1


async def test_process_never_raises_so_a_detached_job_cannot_crash_silently(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
):
    """There is nobody left to hand an exception to; the row records the outcome."""
    ingestor = harness.services.ingestor
    document = await ingestor.create_pending(
        filename="a.md", content_type="text/plain", size_bytes=10, text="du contenu"
    )

    def boom(_texts):
        raise RuntimeError("le backend d'embeddings a explosé")

    monkeypatch.setattr(harness.embedder, "embed_documents", boom)

    await ingestor.process(document["id"], "a.md", "du contenu")  # must not raise

    updated = await harness.db.get_document(document["id"])
    assert updated["status"] == "failed"


async def test_a_document_with_no_usable_content_is_rejected_upfront(harness: Harness):
    """Cheap enough to check while the client is still listening."""
    resp = await harness.client.post(
        "/api/documents/upload", files={"file": ("vide.md", b"   ", "text/markdown")}
    )

    assert resp.status_code == 400
    assert (await harness.client.get("/api/documents")).json() == []


async def test_abandoned_ingestions_are_failed_at_startup(harness: Harness):
    """A restart mid-job cannot resume, so the row must not sit in `processing`."""
    await harness.services.ingestor.create_pending(
        filename="interrompu.md",
        content_type="text/plain",
        size_bytes=10,
        text="du contenu",
    )

    reconciled = await harness.db.fail_stale_processing()

    assert reconciled == 1
    [document] = (await harness.client.get("/api/documents")).json()
    assert document["status"] == "failed"


async def test_concurrent_ingestions_are_bounded(harness: Harness):
    """Each job holds a document in memory and pins a CPU; unbounded jobs would
    exhaust the container long before the rate limiter noticed."""
    in_flight = 0
    peak = 0

    class CountingEmbedder(FakeEmbedder):
        def embed_documents(self, texts):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                return super().embed_documents(texts)
            finally:
                in_flight -= 1

    ingestor = DocumentIngestor(
        harness.db,
        CountingEmbedder(),
        TextChunker(400, 50),
        max_concurrent=2,
    )

    documents = [
        await ingestor.create_pending(
            filename=f"doc{i}.md",
            content_type="text/plain",
            size_bytes=10,
            text=f"contenu numéro {i}",
        )
        for i in range(6)
    ]
    await asyncio.gather(
        *(ingestor.process(d["id"], d["filename"], "contenu") for d in documents)
    )

    assert peak <= 2
    finished = [await harness.db.get_document(d["id"]) for d in documents]
    assert [d["status"] for d in finished] == ["completed"] * 6
