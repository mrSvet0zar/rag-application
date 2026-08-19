"""Database layer against a real PostgreSQL + pgvector.

Several of these pin bugs that actually shipped during development: an index
that silently returned nothing, and a MIME type too long for its column.
"""

from __future__ import annotations

import asyncpg
import pytest

from app.config import Settings
from app.vector_db import Database
from tests.doubles import FakeEmbedder

pytestmark = pytest.mark.usefixtures("db")

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---------- schema bootstrap ----------
async def test_connect_applies_the_schema(db: Database):
    async with db.pool.acquire() as conn:
        extension = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
    assert extension == 1
    assert {"documents", "chunks", "conversations", "messages"} <= {
        t["tablename"] for t in tables
    }


async def test_embedding_index_is_hnsw(db: Database):
    """IVFFlat under-probes on a small table and returned zero rows; HNSW is
    the fix and must not silently regress."""
    async with db.pool.acquire() as conn:
        indexdef = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_chunks_embedding'"
        )
    assert indexdef is not None
    assert "hnsw" in indexdef.lower()


# ---------- documents ----------
async def test_document_lifecycle(db: Database):
    doc_id = await db.create_document("notes.md", "text/markdown", 1234)

    created = await db.get_document(doc_id)
    assert created["status"] == "processing"
    assert created["total_chunks"] == 0
    assert created["file_size_bytes"] == 1234

    await db.finalize_document(doc_id, 7, status="completed")
    finalized = await db.get_document(doc_id)
    assert (finalized["status"], finalized["total_chunks"]) == ("completed", 7)


async def test_get_missing_document_returns_none(db: Database):
    assert await db.get_document(999_999) is None


async def test_long_content_type_is_truncated_not_rejected(db: Database):
    """The docx MIME type is 71 chars; the column holds 50."""
    doc_id = await db.create_document("cv.docx", DOCX_MIME, 10)
    stored = await db.get_document(doc_id)
    assert stored["content_type"] == DOCX_MIME[:50]


async def test_documents_are_listed_newest_first(db: Database):
    first = await db.create_document("a.md", "text/markdown", 1)
    second = await db.create_document("b.md", "text/markdown", 1)
    listed = [d["id"] for d in await db.list_documents()]
    assert listed[:2] == [second, first]


async def test_deleting_a_document_cascades_to_its_chunks(db: Database):
    embedder = FakeEmbedder()
    doc_id = await db.create_document("a.md", "text/markdown", 1)
    texts = ["premier passage", "second passage"]
    await db.store_chunks(
        doc_id,
        [
            (t, e, {"filename": "a.md"})
            for t, e in zip(texts, embedder.embed_documents(texts), strict=False)
        ],
    )

    removed = await db.delete_document(doc_id)

    assert removed == 2
    assert await db.get_document(doc_id) is None
    async with db.pool.acquire() as conn:
        assert await conn.fetchval("SELECT COUNT(*) FROM chunks") == 0


# ---------- chunks & search ----------
async def _index(db: Database, filename: str, texts: list[str]) -> int:
    embedder = FakeEmbedder()
    doc_id = await db.create_document(filename, "text/markdown", 1)
    await db.store_chunks(
        doc_id,
        [
            (t, e, {"filename": filename})
            for t, e in zip(texts, embedder.embed_documents(texts), strict=False)
        ],
    )
    await db.finalize_document(doc_id, len(texts))
    return doc_id


async def test_chunks_are_indexed_in_order(db: Database):
    doc_id = await _index(db, "a.md", ["alpha", "beta", "gamma"])
    async with db.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chunk_index, text FROM chunks WHERE document_id = $1 "
            "ORDER BY chunk_index",
            doc_id,
        )
    assert [r["text"] for r in rows] == ["alpha", "beta", "gamma"]
    assert [r["chunk_index"] for r in rows] == [0, 1, 2]


async def test_duplicate_chunk_index_is_rejected(db: Database):
    """The unique constraint is what stops a retried ingest double-indexing."""
    embedder = FakeEmbedder()
    doc_id = await db.create_document("a.md", "text/markdown", 1)
    chunk: list[tuple[str, list[float], dict]] = [
        ("texte", embedder.embed_query("texte"), {})
    ]
    await db.store_chunks(doc_id, chunk)
    with pytest.raises(asyncpg.UniqueViolationError):
        await db.store_chunks(doc_id, chunk)


async def test_search_ranks_the_closest_chunk_first(db: Database):
    embedder = FakeEmbedder()
    await _index(
        db,
        "guide.md",
        [
            "pgvector index HNSW recherche vectorielle",  # strongest overlap
            "pgvector est une extension de PostgreSQL parmi beaucoup d autres",
            "la tarte aux pommes se cuit vingt minutes au four",  # unrelated
        ],
    )

    results = await db.search(
        embedder.embed_query("pgvector index HNSW"), top_k=5, min_score=0.0
    )

    assert results[0]["text"].startswith("pgvector index HNSW")
    assert results[0]["similarity"] > results[1]["similarity"]
    assert all("tarte" not in r["text"] for r in results)


async def test_search_excludes_chunks_with_zero_similarity(db: Database):
    """`WHERE 1 - (embedding <=> q) > min_score` is a strict comparison, so a
    chunk with nothing in common is dropped even with the floor at 0."""
    embedder = FakeEmbedder()
    await _index(db, "a.md", ["la tarte aux pommes"])

    assert (
        await db.search(embedder.embed_query("kubernetes ingress"), min_score=0.0) == []
    )


async def test_search_joins_the_source_filename(db: Database):
    embedder = FakeEmbedder()
    await _index(db, "guide.md", ["contenu indexé"])
    [hit] = await db.search(embedder.embed_query("contenu indexé"), top_k=1)
    assert hit["filename"] == "guide.md"
    assert set(hit) >= {"id", "document_id", "text", "filename", "similarity"}


async def test_search_respects_top_k(db: Database):
    embedder = FakeEmbedder()
    await _index(db, "a.md", [f"passage numéro {i}" for i in range(6)])
    assert len(await db.search(embedder.embed_query("passage"), top_k=3)) == 3


async def test_search_filters_below_min_score(db: Database):
    embedder = FakeEmbedder()
    await _index(db, "a.md", ["un sujet totalement different des autres"])
    query = embedder.embed_query("sujet different")

    permissive = await db.search(query, min_score=0.0)
    strict = await db.search(query, min_score=0.99)

    assert permissive != [], "the query shares words, so it should match"
    assert strict == [], "a near-1.0 floor must reject a partial match"


async def test_search_on_empty_corpus_returns_nothing(db: Database):
    assert await db.search(FakeEmbedder().embed_query("quoi que ce soit")) == []


# ---------- conversations & messages ----------
async def test_conversation_tracks_messages_and_title(db: Database):
    convo_id = await db.create_conversation()
    assert await db.conversation_exists(convo_id)

    await db.add_message(convo_id, "user", "Quelle est la question ?")
    await db.add_message(convo_id, "assistant", "Voici la réponse.", tokens_used=12)

    convo = await db.get_conversation(convo_id)
    assert convo["total_messages"] == 2
    assert convo["title"] == "Quelle est la question ?"[:60]
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]
    assert convo["last_message_at"] is not None


async def test_message_records_retrieved_chunk_ids(db: Database):
    doc_id = await _index(db, "a.md", ["un passage"])
    async with db.pool.acquire() as conn:
        chunk_id = await conn.fetchval(
            "SELECT id FROM chunks WHERE document_id = $1", doc_id
        )
    convo_id = await db.create_conversation()
    await db.add_message(convo_id, "assistant", "ok", retrieved_chunk_ids=[chunk_id])

    async with db.pool.acquire() as conn:
        stored = await conn.fetchval(
            "SELECT retrieved_chunk_ids FROM messages WHERE conversation_id = $1",
            convo_id,
        )
    assert stored == [chunk_id]


async def test_unknown_conversation(db: Database):
    from uuid import uuid4

    assert await db.get_conversation(uuid4()) is None
    assert await db.conversation_exists(uuid4()) is False


async def test_deleting_a_conversation_cascades_to_messages(db: Database):
    convo_id = await db.create_conversation()
    await db.add_message(convo_id, "user", "salut")
    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE id = $1", convo_id)
        assert await conn.fetchval("SELECT COUNT(*) FROM messages") == 0


# ---------- stats ----------
async def test_stats_counts_everything(db: Database):
    await _index(db, "a.md", ["un", "deux"])
    convo_id = await db.create_conversation()
    await db.add_message(convo_id, "user", "coucou")

    assert await db.get_stats() == {
        "total_documents": 1,
        "total_chunks": 2,
        "total_conversations": 1,
        "total_messages": 1,
    }


# ---------- connection resilience ----------
async def test_connect_gives_up_after_its_budget(test_settings: Settings):
    """Railway's private DNS needs a few seconds; an unreachable host must
    still fail fast rather than hang forever."""
    unreachable = Settings(
        database_url="postgresql://u:p@no-such-host.invalid:5432/db",
        anthropic_api_key="",
    )
    with pytest.raises((OSError, asyncpg.PostgresError)):
        await Database(unreachable).connect(max_wait_seconds=0.2)


# ---------- lexical search ----------
async def test_lexical_search_finds_an_exact_term(db: Database):
    """The case vector search is blind to: a foreign title inside French prose."""
    await _index(
        db,
        "transformeur.md",
        [
            "Le transformeur est décrit dans Attention Is All You Need.",
            "Les réseaux récurrents traitent les séquences pas à pas.",
        ],
    )

    results = await db.search_lexical("Attention Is All You Need", top_k=5)

    assert results[0]["text"].startswith("Le transformeur est décrit")
    assert results[0]["lexical_score"] > 0


async def test_lexical_search_ors_the_query_terms(db: Database):
    """ANDing a natural-language question would match nothing at all: no chunk
    contains every content word of a real question."""
    await _index(db, "a.md", ["Le transformeur est une architecture profonde."])

    results = await db.search_lexical(
        "Quel article scientifique a présenté le transformeur ?", top_k=5
    )

    assert len(results) == 1


async def test_lexical_search_applies_french_stemming(db: Database):
    """Indexed with the french dictionary, so inflections must match."""
    await _index(db, "a.md", ["Les vecteurs normalisés facilitent la recherche."])

    assert await db.search_lexical("vecteur normalisé", top_k=5)


async def test_lexical_search_returns_nothing_for_a_query_without_lexemes(
    db: Database,
):
    """A query of stopwords parses to an empty tsquery; the guard must hold."""
    await _index(db, "a.md", ["Un contenu quelconque."])

    assert await db.search_lexical("le la les de des", top_k=5) == []


async def test_lexical_search_respects_top_k(db: Database):
    await _index(db, "a.md", [f"pgvector chunk numéro {i}" for i in range(6)])

    assert len(await db.search_lexical("pgvector", top_k=3)) == 3


async def test_lexical_search_joins_the_source_filename(db: Database):
    await _index(db, "guide.md", ["indexation lexicale du contenu"])

    [hit] = await db.search_lexical("indexation lexicale", top_k=1)

    assert hit["filename"] == "guide.md"
    assert set(hit) >= {"id", "document_id", "text", "filename", "lexical_score"}
