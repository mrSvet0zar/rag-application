"""Unit tests that don't require the database or the embedding model download."""

import asyncio

import pytest

from app.config import Settings
from app.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    """A pipeline forced into demo mode (no API key), independent of .env."""
    return RAGPipeline(settings=Settings(anthropic_api_key=""))


def test_cors_origins_parsing():
    s = Settings(cors_origins="http://a.com, http://b.com ,")
    assert s.cors_origins_list == ["http://a.com", "http://b.com"]


def test_demo_mode_toggle():
    assert Settings(anthropic_api_key="").demo_mode is True
    assert Settings(anthropic_api_key="sk-ant-xyz").demo_mode is False


def test_split_text_produces_chunks(pipeline):
    text = "Phrase un. " * 500  # long enough to split
    chunks = pipeline.split_text(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_split_text_empty(pipeline):
    assert pipeline.split_text("   ") == []


def test_build_context_includes_sources():
    chunks = [
        {"text": "Contenu A", "filename": "a.md", "document_id": 1},
        {"text": "Contenu B", "filename": "b.md", "document_id": 2},
    ]
    context = RAGPipeline._build_context(chunks)
    assert "a.md" in context and "b.md" in context
    assert "Contenu A" in context and "Contenu B" in context


def test_generate_response_no_chunks(pipeline):
    answer, tokens = asyncio.run(pipeline.generate_response("Question ?", []))
    assert tokens == 0
    assert "aucun passage" in answer.lower()


def test_demo_answer_mentions_sources(pipeline):
    chunks = [{"text": "Un extrait pertinent.", "filename": "doc.md", "document_id": 1}]
    answer, tokens = asyncio.run(pipeline.generate_response("Question ?", chunks))
    assert tokens == 0
    assert "démo" in answer.lower()
    assert "doc.md" in answer


def test_reranker_assemble_sorts_and_truncates():
    """_assemble adds sigmoid scores, sorts by relevance, keeps top_n."""
    from app.reranker import RerankService

    chunks = [
        {"id": 1, "text": "peu pertinent"},
        {"id": 2, "text": "très pertinent"},
        {"id": 3, "text": "moyen"},
    ]
    raw_scores = [-2.0, 5.0, 0.0]  # cross-encoder logits
    out = RerankService._assemble(chunks, raw_scores, top_n=2)

    assert [c["id"] for c in out] == [2, 3]  # highest logits first
    assert 0.0 <= out[0]["rerank_score"] <= 1.0
    assert out[0]["rerank_score"] > out[1]["rerank_score"]
    # original chunks are not mutated (dicts are copied)
    assert "rerank_score" not in chunks[0]


def test_reranker_assemble_empty():
    from app.reranker import RerankService

    assert RerankService._assemble([], [], top_n=5) == []


def test_reranker_assemble_threshold_filters_but_keeps_best():
    from app.reranker import RerankService

    chunks = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    scores = [5.0, -6.0, -7.0]  # only #1 is clearly relevant
    out = RerankService._assemble(chunks, scores, top_n=5, min_score=0.05)
    assert [c["id"] for c in out] == [1]  # low-score chunks dropped

    # If everything is below threshold, still keep the single best.
    out2 = RerankService._assemble(chunks, [-6.0, -7.0, -8.0], top_n=5, min_score=0.05)
    assert len(out2) == 1 and out2[0]["id"] == 1


def test_llm_error_answer_credit_message():
    """The credit-balance error produces a clear, user-facing fallback."""
    import anthropic

    exc = anthropic.APIError.__new__(anthropic.APIError)
    exc.message = "Your credit balance is too low to access the Anthropic API."
    chunks = [{"text": "Extrait.", "filename": "doc.md", "document_id": 1}]
    answer = RAGPipeline._llm_error_answer("Q ?", chunks, exc)
    assert "crédits" in answer.lower()
    assert "doc.md" in answer
