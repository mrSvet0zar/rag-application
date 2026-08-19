"""End-to-end HTTP tests: real routing, real database, fake models."""

from __future__ import annotations

import io
from uuid import uuid4

import pytest

from tests.integration.conftest import Harness


def sse_events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event, payload) pairs."""
    import json

    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        event, data = "message", ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((event, json.loads(data) if data else {}))
    return events


async def upload(harness: Harness, name: str, body: bytes, mime: str = "text/markdown"):
    return await harness.client.post(
        "/api/documents/upload", files={"file": (name, body, mime)}
    )


# ---------- health & stats ----------
async def test_health(harness: Harness):
    resp = await harness.client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "demo_mode": True}


async def test_stats_reflect_ingested_content(harness: Harness):
    await upload(harness, "a.md", b"un contenu de test suffisant")

    body = (await harness.client.get("/api/stats")).json()

    assert body["total_documents"] == 1
    assert body["total_chunks"] >= 1
    assert body["demo_mode"] is True
    assert body["rerank_enabled"] is False


# ---------- upload ----------
async def test_upload_is_accepted_immediately(harness: Harness):
    """202 with the row still `processing`: the client gets a receipt, not a wait."""
    resp = await upload(harness, "guide.md", b"pgvector et HNSW")

    assert resp.status_code == 202
    body = resp.json()
    assert body["filename"] == "guide.md"
    assert body["status"] == "processing"
    assert body["file_size_bytes"] == len(b"pgvector et HNSW")


async def test_upload_indexes_the_document_in_the_background(harness: Harness):
    resp = await upload(harness, "guide.md", b"pgvector et HNSW")

    document = (await harness.client.get(f"/api/documents/{resp.json()['id']}")).json()

    assert document["status"] == "completed"
    assert document["total_chunks"] == 1


async def test_upload_splits_long_documents(harness: Harness):
    resp = await upload(harness, "long.md", ("Phrase de test. " * 300).encode())

    document = (await harness.client.get(f"/api/documents/{resp.json()['id']}")).json()

    assert document["total_chunks"] > 1


async def test_upload_rejects_an_empty_file(harness: Harness):
    resp = await upload(harness, "vide.md", b"")
    assert resp.status_code == 400
    assert "detail" in resp.json()


async def test_upload_rejects_whitespace_only_content(harness: Harness):
    resp = await upload(harness, "blanc.md", b"   \n\t  ")
    assert resp.status_code == 400


async def test_upload_accepts_docx(harness: Harness):
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph("Le contenu du document Word de test.")
    buffer = io.BytesIO()
    doc.save(buffer)

    resp = await upload(
        harness,
        "cv.docx",
        buffer.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert resp.status_code == 202
    document = (await harness.client.get(f"/api/documents/{resp.json()['id']}")).json()
    assert document["total_chunks"] == 1


async def test_upload_failure_marks_the_document_failed(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
):
    """A crash mid-ingest must not leave a row stuck in `processing`."""

    def boom(_texts):
        raise RuntimeError("embedding backend exploded")

    monkeypatch.setattr(harness.embedder, "embed_documents", boom)

    resp = await upload(harness, "a.md", b"du contenu")

    # Accepted, because the failure happens after the client has been answered.
    assert resp.status_code == 202
    [document] = (await harness.client.get("/api/documents")).json()
    assert document["status"] == "failed", "jamais laissé en `processing`"


# ---------- listing & deletion ----------
async def test_list_and_delete_documents(harness: Harness):
    created = (await upload(harness, "a.md", b"contenu a")).json()
    await upload(harness, "b.md", b"contenu b")

    listed = (await harness.client.get("/api/documents")).json()
    assert {d["filename"] for d in listed} == {"a.md", "b.md"}

    deleted = await harness.client.delete(f"/api/documents/{created['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["chunks_deleted"] == 1

    remaining = (await harness.client.get("/api/documents")).json()
    assert [d["filename"] for d in remaining] == ["b.md"]


async def test_delete_unknown_document_is_404(harness: Harness):
    resp = await harness.client.delete("/api/documents/424242")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Document not found."}


# ---------- URL import ----------
async def test_import_url_refuses_a_private_address(harness: Harness):
    """Proves the SSRF guard is actually wired into the endpoint."""
    resp = await harness.client.post(
        "/api/documents/import-url", json={"url": "http://127.0.0.1:8000/admin"}
    )
    assert resp.status_code == 400
    assert "refus" in resp.json()["detail"].lower()


async def test_import_url_rejects_a_non_http_scheme(harness: Harness):
    resp = await harness.client.post(
        "/api/documents/import-url", json={"url": "ftp://example.com/x"}
    )
    assert resp.status_code == 422  # rejected by the HttpUrl schema


class _FakeResponse:
    """Mimics the streamed response the endpoint consumes.

    The body is served in small pieces so the test exercises the same
    incremental read path as a real download.
    """

    def __init__(self, html: str) -> None:
        self._body = html.encode()
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to do
        pass

    async def aiter_bytes(self):
        for start in range(0, len(self._body), 16):
            yield self._body[start : start + 16]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, html: str) -> None:
    """Bypass the network (and the DNS-based guard) for endpoint-level tests."""
    from app.api import documents as documents_module

    monkeypatch.setattr(documents_module, "validate_public_url", lambda _url: None)

    class _FakeClient:
        def __init__(self, **_kwargs) -> None: ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url, **_kwargs):
            # Not async: httpx.stream returns an async context manager, it is
            # not awaited itself.
            return _FakeResponse(html)

    monkeypatch.setattr(documents_module.httpx, "AsyncClient", _FakeClient)


async def test_import_url_titles_the_document_from_the_page(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
):
    _patch_fetch(
        monkeypatch,
        "<html><head><title>Guide pgvector</title></head>"
        "<body><script>ignore()</script><p>Un contenu utile.</p></body></html>",
    )

    resp = await harness.client.post(
        "/api/documents/import-url", json={"url": "https://example.com/guide"}
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["filename"] == "Guide pgvector"
    assert body["content_type"] == "text/html"

    document = (await harness.client.get(f"/api/documents/{body['id']}")).json()
    assert document["total_chunks"] == 1


async def test_import_url_rejects_a_page_without_text(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
):
    _patch_fetch(monkeypatch, "<html><head><title>x</title></head><body></body></html>")
    resp = await harness.client.post(
        "/api/documents/import-url", json={"url": "https://example.com/empty"}
    )
    assert resp.status_code == 400


# ---------- chat ----------
async def test_chat_answers_and_persists_the_exchange(harness: Harness):
    await upload(harness, "guide.md", b"pgvector propose un index HNSW")

    resp = await harness.client.post(
        "/api/chat", json={"question": "index HNSW pgvector", "k": 3}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == harness.generator.answer
    assert body["tokens_used"] == harness.generator.tokens
    assert body["processing_time_ms"] >= 0
    assert body["retrieved_chunks"], "the question matches an indexed chunk"
    assert body["retrieved_chunks"][0]["filename"] == "guide.md"
    assert body["retrieved_chunks"][0]["rerank_score"] is None

    convo = (
        await harness.client.get(f"/api/conversations/{body['conversation_id']}")
    ).json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]
    assert convo["messages"][1]["content"] == harness.generator.answer


async def test_chat_passes_retrieved_context_to_the_generator(harness: Harness):
    await upload(harness, "guide.md", "le contenu indexé de référence".encode())

    await harness.client.post("/api/chat", json={"question": "contenu indexé"})

    question, chunks = harness.generator.calls[-1]
    assert question == "contenu indexé"
    assert chunks and "référence" in chunks[0]["text"]


async def test_chat_without_documents_still_answers(harness: Harness):
    resp = await harness.client.post("/api/chat", json={"question": "et alors ?"})
    assert resp.status_code == 200
    assert resp.json()["retrieved_chunks"] == []


async def test_chat_continues_an_existing_conversation(harness: Harness):
    first = (await harness.client.post("/api/chat", json={"question": "un"})).json()

    second = (
        await harness.client.post(
            "/api/chat",
            json={"question": "deux", "conversation_id": first["conversation_id"]},
        )
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    convo = (
        await harness.client.get(f"/api/conversations/{first['conversation_id']}")
    ).json()
    assert len(convo["messages"]) == 4


async def test_chat_with_an_unknown_conversation_starts_a_new_one(harness: Harness):
    """A client holding a stale id should keep working, not get a 404."""
    stale = str(uuid4())
    body = (
        await harness.client.post(
            "/api/chat", json={"question": "salut", "conversation_id": stale}
        )
    ).json()
    assert body["conversation_id"] != stale


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x", "k": 0}])
async def test_chat_rejects_invalid_payloads(harness: Harness, payload: dict):
    resp = await harness.client.post("/api/chat", json=payload)
    assert resp.status_code == 422


@pytest.mark.rerank
async def test_chat_exposes_rerank_scores_when_enabled(harness: Harness):
    await upload(harness, "a.md", b"premier passage\n\nsecond passage")

    body = (
        await harness.client.post("/api/chat", json={"question": "passage", "k": 5})
    ).json()

    scores = [c["rerank_score"] for c in body["retrieved_chunks"]]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


# ---------- streaming ----------
async def test_chat_stream_emits_sources_tokens_then_done(harness: Harness):
    await upload(harness, "guide.md", b"pgvector et son index HNSW")

    resp = await harness.client.post(
        "/api/chat/stream", json={"question": "index HNSW", "k": 3}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = sse_events(resp.text)
    kinds = [name for name, _ in events]
    assert kinds[0] == "sources"
    assert kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"}

    sources = events[0][1]["retrieved_chunks"]
    assert sources and sources[0]["filename"] == "guide.md"

    streamed = "".join(payload["text"] for name, payload in events if name == "token")
    assert streamed.strip() == harness.generator.answer

    done = events[-1][1]
    assert done["tokens_used"] == harness.generator.tokens
    convo = (
        await harness.client.get(f"/api/conversations/{done['conversation_id']}")
    ).json()
    assert convo["messages"][-1]["content"].strip() == harness.generator.answer


async def test_chat_stream_reports_failures_as_an_error_event(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
):
    """The response has already begun, so errors can only travel in-band."""

    async def boom(*_args, **_kwargs):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr(harness.services.retriever, "retrieve", boom)

    resp = await harness.client.post("/api/chat/stream", json={"question": "q"})

    assert resp.status_code == 200
    name, payload = sse_events(resp.text)[-1]
    assert name == "error"
    assert "exploded" in payload["detail"]


# ---------- conversations ----------
async def test_unknown_conversation_is_404(harness: Harness):
    resp = await harness.client.get(f"/api/conversations/{uuid4()}")
    assert resp.status_code == 404


async def test_malformed_conversation_id_is_422(harness: Harness):
    assert (await harness.client.get("/api/conversations/not-a-uuid")).status_code == 422
