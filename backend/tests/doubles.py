"""Test doubles for the injected services.

The real embedder pulls ~500 MB of torch weights and the reranker another
model; loading either would make the suite unusable in CI. These stand-ins are
deterministic, instant, and — crucially for the integration tests — produce
*meaningful* similarity, so retrieval assertions stay honest.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncIterator

DIMENSION = 384
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _bucket(token: str) -> int:
    """Stable bucket for a token (hash() is salted per process, md5 is not)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % DIMENSION


class FakeEmbedder:
    """Hashed bag-of-words embedder.

    Not semantic, but lexically meaningful and L2-normalised, so cosine
    similarity behaves like the real thing for texts that share words.
    """

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.documents: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return DIMENSION

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIMENSION
        for token in _TOKEN_RE.findall(text.lower()):
            vec[_bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # pgvector's cosine distance is undefined for the zero vector, so
            # never emit one.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return self._vector(query)


class RecordingVectorStore:
    """Returns canned rows and records how it was queried."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[dict] = []

    async def search(
        self, query_embedding: list[float], top_k: int, min_score: float
    ) -> list[dict]:
        self.calls.append({"top_k": top_k, "min_score": min_score})
        return list(self._rows)


class ReversingReranker:
    """Deterministic reranker: reverses the candidate order.

    Reversing (rather than preserving) order makes it obvious in assertions
    whether the reranker's ranking actually won.
    """

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        self.calls += 1
        reversed_chunks = list(reversed(chunks))
        out = []
        for rank, chunk in enumerate(reversed_chunks):
            enriched = dict(chunk)
            enriched["rerank_score"] = 1.0 - rank * 0.1
            out.append(enriched)
        return out[:top_n]


class StubGenerator:
    """Canned answer, no network. Records what context it was given."""

    def __init__(self, answer: str = "Réponse de test.", tokens: int = 42) -> None:
        self.answer = answer
        self.tokens = tokens
        self.calls: list[tuple[str, list[dict]]] = []

    async def generate_response(self, query: str, chunks: list[dict]) -> tuple[str, int]:
        self.calls.append((query, chunks))
        return self.answer, self.tokens

    async def stream_response(
        self, query: str, chunks: list[dict]
    ) -> AsyncIterator[tuple[str, object]]:
        self.calls.append((query, chunks))
        for word in self.answer.split(" "):
            yield ("token", word + " ")
        yield ("usage", self.tokens)
