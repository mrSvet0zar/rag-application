"""Document ingestion: text -> chunks -> embeddings -> stored rows.

Ingestion is split in two so it can run outside the request that triggered it.
Chunking and embedding a large PDF takes far longer than a client is willing to
hold a connection open, and doing it inline means the upload either times out or
blocks a worker for the duration. `create_pending` records the document
immediately; `process` does the expensive part afterwards and moves the row from
`processing` to `completed` or `failed`.
"""

from __future__ import annotations

import asyncio
import logging

from app.chunking import TextChunker
from app.errors import IngestionFailedError, UnreadableDocumentError
from app.protocols import Embedder
from app.vector_db import Database

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Turns extracted text into an indexed document.

    Shared by file upload and URL import so both paths behave identically,
    including the failure path: a document is never left dangling in
    `processing` because something threw.
    """

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        chunker: TextChunker,
        max_concurrent: int = 2,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._chunker = chunker
        # Embedding is CPU-bound and each job holds a whole document in memory.
        # Unbounded background jobs would let a handful of simultaneous uploads
        # exhaust the container long before the rate limiter noticed.
        self._slots = asyncio.Semaphore(max_concurrent)

    async def create_pending(
        self, *, filename: str, content_type: str, size_bytes: int, text: str
    ) -> dict:
        """Record the document as `processing` and return its row.

        Chunking happens here rather than in the background job: it is cheap,
        and a document that produces nothing usable should be rejected while the
        client is still listening instead of failing silently later.
        """
        if not self._chunker.split(text):
            raise UnreadableDocumentError("Document produced no chunks.")

        document_id = await self._db.create_document(filename, content_type, size_bytes)
        document = await self._db.get_document(document_id)
        assert document is not None  # just inserted
        return document

    async def process(self, document_id: int, filename: str, text: str) -> None:
        """Chunk, embed and store. Marks the document completed or failed.

        Never raises: it runs detached from any request, so there is nobody left
        to hand an exception to. The outcome is recorded on the row instead,
        which is what the client polls.
        """
        async with self._slots:
            try:
                chunks = self._chunker.split(text)
                # Embedding is CPU-bound -> keep it off the event loop.
                embeddings = await asyncio.to_thread(
                    self._embedder.embed_documents, chunks
                )
                # strict=True: an embedder returning a different count than it
                # was given is a bug, and silently dropping chunks loses content.
                chunk_data = [
                    (chunk, embedding, {"filename": filename})
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ]
                stored = await self._db.store_chunks(document_id, chunk_data)
                await self._db.finalize_document(document_id, stored, status="completed")
                logger.info(
                    "document indexed",
                    extra={
                        "event": "ingestion.completed",
                        "document_id": document_id,
                        "chunks": stored,
                    },
                )
            except Exception:
                await self._db.finalize_document(document_id, 0, status="failed")
                logger.exception(
                    "document ingestion failed",
                    extra={
                        "event": "ingestion.failed",
                        "document_id": document_id,
                    },
                )

    async def ingest(
        self, *, filename: str, content_type: str, size_bytes: int, text: str
    ) -> dict:
        """Index `text` end to end and return the finished row.

        The synchronous path, for callers that want the result rather than a
        receipt: the evaluation harness and the seeding script. The API uses
        `create_pending` + `process` instead.
        """
        document = await self.create_pending(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            text=text,
        )
        await self.process(document["id"], filename, text)

        finished = await self._db.get_document(document["id"])
        assert finished is not None
        if finished["status"] == "failed":
            raise IngestionFailedError(f"Indexation de {filename!r} échouée.")
        return finished
