"""Composition root.

Everything the app needs is constructed here, once, and handed to the API layer
through `app.state`. Nothing imports a module-level singleton, which is what
makes the whole stack substitutable in tests (fake embedder, stub generator,
throwaway database) without monkeypatching.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking import TextChunker
from app.config import Settings
from app.embeddings import EmbeddingService
from app.generation import AnswerGenerator
from app.ingestor import DocumentIngestor
from app.protocols import Embedder, Generator, Reranker
from app.reranker import RerankService
from app.retrieval import Retriever
from app.vector_db import Database


@dataclass(frozen=True)
class Services:
    """The application's wired object graph."""

    settings: Settings
    db: Database
    embedder: Embedder
    reranker: Reranker | None
    chunker: TextChunker
    generator: Generator
    retriever: Retriever
    ingestor: DocumentIngestor


def build_services(settings: Settings) -> Services:
    """Wire the production object graph (no I/O yet — see `Database.connect`)."""
    db = Database(settings)
    embedder: Embedder = EmbeddingService(settings)
    # A disabled reranker is represented as None rather than a flag checked
    # deep in the retrieval code.
    reranker: Reranker | None = (
        RerankService(settings) if settings.rerank_enabled else None
    )
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    generator: Generator = AnswerGenerator(settings)

    return Services(
        settings=settings,
        db=db,
        embedder=embedder,
        reranker=reranker,
        chunker=chunker,
        generator=generator,
        retriever=Retriever(db, embedder, reranker, settings),
        ingestor=DocumentIngestor(db, embedder, chunker),
    )
