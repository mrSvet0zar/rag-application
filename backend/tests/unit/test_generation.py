import anthropic
import pytest

from app.config import Settings
from app.generation import AnswerGenerator


@pytest.fixture
def generator() -> AnswerGenerator:
    """Demo-mode generator (no API key), independent of the ambient .env."""
    return AnswerGenerator(Settings(anthropic_api_key=""))


def test_build_context_includes_sources():
    chunks = [
        {"text": "Contenu A", "filename": "a.md", "document_id": 1},
        {"text": "Contenu B", "filename": "b.md", "document_id": 2},
    ]
    context = AnswerGenerator._build_context(chunks)
    assert "a.md" in context and "b.md" in context
    assert "Contenu A" in context and "Contenu B" in context


def test_build_context_falls_back_to_document_id():
    context = AnswerGenerator._build_context(
        [{"text": "x", "filename": None, "document_id": 7}]
    )
    assert "document #7" in context


async def test_generate_response_without_chunks(generator):
    answer, tokens = await generator.generate_response("Question ?", [])
    assert tokens == 0
    assert "aucun passage" in answer.lower()


async def test_demo_answer_mentions_sources(generator):
    chunks = [{"text": "Un extrait pertinent.", "filename": "doc.md", "document_id": 1}]
    answer, tokens = await generator.generate_response("Question ?", chunks)
    assert tokens == 0
    assert "démo" in answer.lower()
    assert "doc.md" in answer


async def test_stream_response_in_demo_mode_yields_tokens_then_usage(generator):
    chunks = [{"text": "Extrait.", "filename": "doc.md", "document_id": 1}]
    events = [e async for e in generator.stream_response("Q ?", chunks)]

    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "usage", "usage must be the terminal event"
    assert kinds.count("usage") == 1
    assert all(k == "token" for k in kinds[:-1])
    assert "démo" in "".join(str(d) for k, d in events if k == "token").lower()


async def test_stream_response_without_chunks(generator):
    events = [e async for e in generator.stream_response("Q ?", [])]
    assert [k for k, _ in events] == ["token", "usage"]
    assert "aucun passage" in str(events[0][1]).lower()


def _api_error(message: str) -> anthropic.APIError:
    exc = anthropic.APIError.__new__(anthropic.APIError)
    exc.message = message
    return exc


def test_llm_error_answer_explains_credit_exhaustion():
    chunks = [{"text": "Extrait.", "filename": "doc.md", "document_id": 1}]
    answer = AnswerGenerator._llm_error_answer(
        "Q ?", chunks, _api_error("Your credit balance is too low to access the API.")
    )
    assert "crédits" in answer.lower()
    assert "doc.md" in answer, "the fallback must still surface the retrieved context"


def test_llm_error_answer_generic_reason_is_reported():
    chunks = [{"text": "Extrait.", "filename": "doc.md", "document_id": 1}]
    answer = AnswerGenerator._llm_error_answer("Q ?", chunks, _api_error("boom"))
    assert "boom" in answer
