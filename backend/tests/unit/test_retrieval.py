"""Two-stage retrieval logic, exercised with doubles (no DB, no models)."""

import pytest

from app.config import Settings
from app.retrieval import Retriever
from tests.doubles import FakeEmbedder, RecordingVectorStore, ReversingReranker

ROWS = [
    {"id": 1, "text": "a", "document_id": 1, "filename": "a.md", "similarity": 0.9},
    {"id": 2, "text": "b", "document_id": 1, "filename": "a.md", "similarity": 0.5},
    {"id": 3, "text": "c", "document_id": 2, "filename": "b.md", "similarity": 0.2},
]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="",
        min_relevance_score=0.25,
        rerank_candidates=20,
    )


async def test_without_reranker_uses_k_and_the_similarity_floor(settings):
    store = RecordingVectorStore(ROWS)
    retriever = Retriever(store, FakeEmbedder(), None, settings)

    result = await retriever.retrieve("question", k=2)

    assert store.calls == [{"top_k": 2, "min_score": 0.25}]
    assert result == ROWS  # store decides; retriever must not re-filter


async def test_with_reranker_fetches_a_broad_pool_with_no_floor(settings):
    """The cross-encoder must see low-cosine candidates; a floor would hide
    them (a 0.19-cosine chunk has been observed reranking to #1)."""
    store = RecordingVectorStore(ROWS)
    retriever = Retriever(store, FakeEmbedder(), ReversingReranker(), settings)

    result = await retriever.retrieve("question", k=2)

    assert store.calls == [{"top_k": 20, "min_score": 0.0}]
    assert [c["id"] for c in result] == [3, 2], "reranker ordering must win"


async def test_empty_candidate_pool_skips_the_reranker(settings):
    reranker = ReversingReranker()
    retriever = Retriever(RecordingVectorStore([]), FakeEmbedder(), reranker, settings)

    assert await retriever.retrieve("question", k=5) == []
    assert reranker.calls == 0, "no point paying for a rerank of nothing"


async def test_query_is_embedded_once_per_retrieval(settings):
    embedder = FakeEmbedder()
    retriever = Retriever(RecordingVectorStore(ROWS), embedder, None, settings)

    await retriever.retrieve("question", k=1)

    assert embedder.queries == ["question"]
