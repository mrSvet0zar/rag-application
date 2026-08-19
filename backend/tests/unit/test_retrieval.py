"""Retrieval logic — candidate generation, fusion, reranking — exercised with
doubles (no database, no models)."""

import pytest

from app.config import Settings
from app.retrieval import Retriever
from tests.doubles import FakeEmbedder, RecordingVectorStore, ReversingReranker

ROWS = [
    {"id": 1, "text": "a", "document_id": 1, "filename": "a.md", "similarity": 0.9},
    {"id": 2, "text": "b", "document_id": 1, "filename": "a.md", "similarity": 0.5},
    {"id": 3, "text": "c", "document_id": 2, "filename": "b.md", "similarity": 0.2},
]


LEXICAL_ROWS = [
    {"id": 9, "text": "z", "document_id": 3, "filename": "c.md", "lexical_score": 0.7},
    {"id": 2, "text": "b", "document_id": 1, "filename": "a.md", "lexical_score": 0.4},
]


@pytest.fixture
def settings() -> Settings:
    """Dense-only by default; hybrid tests opt in explicitly."""
    return Settings(
        anthropic_api_key="",
        min_relevance_score=0.25,
        rerank_candidates=20,
        hybrid_enabled=False,
    )


async def test_dense_only_without_reranker_uses_k_and_the_similarity_floor(settings):
    store = RecordingVectorStore(ROWS)
    retriever = Retriever(store, FakeEmbedder(), None, settings)

    result = await retriever.retrieve("question", k=2)

    assert store.calls == [{"top_k": 2, "min_score": 0.25}]
    # Ordering is the store's; the cap is the retriever's, so a caller asking
    # for k never has to trim the result itself.
    assert result == ROWS[:2]


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


async def test_hybrid_queries_both_retrievers_and_fuses_them(settings):
    hybrid = settings.model_copy(update={"hybrid_enabled": True})
    store = RecordingVectorStore(ROWS, LEXICAL_ROWS)
    retriever = Retriever(store, FakeEmbedder(), None, hybrid)

    result = await retriever.retrieve("question", k=5)

    assert store.lexical_calls == [{"query": "question", "top_k": 20}]
    # id 2 is the only chunk both retrievers found, so fusion promotes it.
    assert result[0]["id"] == 2
    # Chunks seen by a single retriever still survive.
    assert {row["id"] for row in result} == {1, 2, 3, 9}


async def test_fusion_keeps_the_scores_each_retriever_contributed(settings):
    hybrid = settings.model_copy(update={"hybrid_enabled": True})
    retriever = Retriever(
        RecordingVectorStore(ROWS, LEXICAL_ROWS), FakeEmbedder(), None, hybrid
    )

    result = await retriever.retrieve("question", k=5)
    by_id = {row["id"]: row for row in result}

    assert by_id[2]["similarity"] == 0.5 and by_id[2]["lexical_score"] == 0.4
    assert "lexical_score" not in by_id[1]  # dense-only hit
    assert "similarity" not in by_id[9]  # lexical-only hit


async def test_hybrid_widens_the_pool_even_without_a_reranker(settings):
    """Fusion needs candidates to fuse; k results would leave nothing to merge."""
    hybrid = settings.model_copy(update={"hybrid_enabled": True})
    store = RecordingVectorStore(ROWS, LEXICAL_ROWS)

    await Retriever(store, FakeEmbedder(), None, hybrid).retrieve("question", k=2)

    assert store.calls == [{"top_k": 20, "min_score": 0.0}]


async def test_hybrid_falls_back_to_dense_when_nothing_matches_lexically(settings):
    """A query of stopwords matches no lexeme; fusing one list is pointless."""
    hybrid = settings.model_copy(update={"hybrid_enabled": True})
    store = RecordingVectorStore(ROWS, [])

    result = await Retriever(store, FakeEmbedder(), None, hybrid).retrieve("q", k=3)

    assert [row["id"] for row in result] == [1, 2, 3]
