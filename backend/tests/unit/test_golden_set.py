"""The golden set is only worth what its snippets are worth.

If a snippet stops matching the corpus — a re-pinned revision, an edited
question, a change in the extraction pipeline — every metric silently drops and
the cause is invisible in the numbers. These tests turn that into a loud
failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.chunking import TextChunker

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_SET = EVAL_DIR / "golden_set.json"

# Snippets must fit inside the overlap between two consecutive chunks;
# otherwise one could straddle a boundary and belong to no chunk at all.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200


def _questions() -> list[dict]:
    return json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]


def _corpus_text(document: str) -> str:
    return (CORPUS_DIR / f"{document}.txt").read_text(encoding="utf-8")


def test_the_golden_set_is_not_empty():
    assert len(_questions()) >= 30


def test_question_ids_are_unique():
    ids = [q["id"] for q in _questions()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("question", _questions(), ids=lambda q: q["id"])
def test_every_snippet_appears_verbatim_in_its_document(question):
    text = _corpus_text(question["document"])
    for snippet in question["snippets"]:
        assert snippet in text, (
            f"{question['id']}: extrait absent de {question['document']}.txt — "
            "le corpus et le golden set ont divergé"
        )


@pytest.mark.parametrize("question", _questions(), ids=lambda q: q["id"])
def test_every_snippet_survives_chunking(question):
    """A snippet split across two chunks would be findable in neither."""
    chunker = TextChunker(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunker.split(_corpus_text(question["document"]))
    for snippet in question["snippets"]:
        assert len(snippet) <= CHUNK_OVERLAP
        assert any(
            snippet in chunk for chunk in chunks
        ), f"{question['id']}: extrait perdu au découpage"


def test_both_question_kinds_are_represented():
    """The lexical/semantic split is what makes the hybrid comparison readable."""
    kinds = [q["kind"] for q in _questions()]
    assert kinds.count("lexical") >= 5
    assert kinds.count("semantic") >= 5


@pytest.mark.parametrize("question", _questions(), ids=lambda q: q["id"])
def test_documents_referenced_by_the_golden_set_exist(question):
    assert (CORPUS_DIR / f"{question['document']}.txt").exists()
