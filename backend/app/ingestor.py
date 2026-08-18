"""Document ingestion: text -> chunks -> embeddings -> stored rows."""

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
    including the failure path (the document row is marked `failed` rather
    than left dangling in `processing`).
    """

    def __init__(self, db: Database, embedder: Embedder, chunker: TextChunker) -> None:
        self._db = db
        self._embedder = embedder
        self._chunker = chunker

    async def ingest(
        self, *, filename: str, content_type: str, size_bytes: int, text: str
    ) -> dict:
        """Index `text` as a document and return the stored document row."""
        chunks = self._chunker.split(text)
        if not chunks:
            raise UnreadableDocumentError("Document produced no chunks.")

        document_id = await self._db.create_document(filename, content_type, size_bytes)
        try:
            # Embedding is CPU-bound -> keep it off the event loop.
            embeddings = await asyncio.to_thread(self._embedder.embed_documents, chunks)
            # strict=True: an embedder returning a different count than it was
            # given is a bug, and silently dropping chunks would lose content.
            chunk_data = [
                (chunk, embedding, {"filename": filename})
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            stored = await self._db.store_chunks(document_id, chunk_data)
            await self._db.finalize_document(document_id, stored, status="completed")
        except Exception as exc:
            await self._db.finalize_document(document_id, 0, status="failed")
            logger.exception("Failed to process document %s", document_id)
            raise IngestionFailedError(str(exc)) from exc

        document = await self._db.get_document(document_id)
        assert document is not None  # just created and finalized
        return document
