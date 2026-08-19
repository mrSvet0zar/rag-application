"""Structural interfaces for the swappable pieces of the RAG pipeline.

Using `Protocol` rather than base classes keeps the concrete services free of
inheritance, while still letting mypy verify that test doubles (a deterministic
fake embedder, a stub generator) are valid substitutes for the real thing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class Embedder(Protocol):
    """Turns text into vectors. Implemented by the local sentence-transformers
    service in production, and by a deterministic fake in tests."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class Reranker(Protocol):
    """Re-scores candidate chunks against the query, best first."""

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]: ...


class VectorStore(Protocol):
    """Similarity search over stored chunks, dense and lexical.

    Both live behind one interface because they are two views of the same
    store — in this implementation, two queries against the same table.
    """

    async def search(
        self, query_embedding: list[float], top_k: int, min_score: float
    ) -> list[dict]: ...

    async def search_lexical(self, query: str, top_k: int) -> list[dict]: ...


class Generator(Protocol):
    """Produces the final answer from the question and retrieved context."""

    async def generate_response(
        self, query: str, chunks: list[dict]
    ) -> tuple[str, int]: ...

    def stream_response(
        self, query: str, chunks: list[dict]
    ) -> AsyncIterator[tuple[str, object]]: ...
