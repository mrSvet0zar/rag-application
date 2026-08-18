"""Document endpoints: upload, URL import, list, delete."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, File, UploadFile

from app.deps import DbDep, IngestorDep
from app.errors import (
    NotFoundError,
    UnreadableDocumentError,
    UrlFetchError,
    UrlNotAllowedError,
)
from app.ingestion import extract_text, html_to_text, validate_public_url
from app.schemas import DocumentResponse, UrlImportRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Cap on how much of a fetched web page we'll read (bytes).
MAX_URL_BYTES = 5 * 1024 * 1024


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    ingestor: IngestorDep, file: UploadFile = File(...)
) -> DocumentResponse:
    """Upload a document (txt/md/pdf/docx/html): extract, chunk, embed, store."""
    content = await file.read()
    if not content:
        raise UnreadableDocumentError("Empty file.")

    filename = file.filename or "document"
    try:
        text = extract_text(filename, content)
    except Exception as exc:
        raise UnreadableDocumentError(f"Could not read file: {exc}") from exc

    if not text.strip():
        raise UnreadableDocumentError("No extractable text in file.")

    document = await ingestor.ingest(
        filename=filename,
        content_type=file.content_type or "text/plain",
        size_bytes=len(content),
        text=text,
    )
    return DocumentResponse(**document)


@router.post("/import-url", response_model=DocumentResponse)
async def import_url(req: UrlImportRequest, ingestor: IngestorDep) -> DocumentResponse:
    """Fetch a web page, extract its readable text, and index it."""
    url = str(req.url)

    # SSRF guard (blocks localhost / private / metadata addresses). DNS
    # resolution is blocking, hence the thread.
    try:
        await asyncio.to_thread(validate_public_url, url)
    except ValueError as exc:
        raise UrlNotAllowedError(f"URL refusée : {exc}") from exc

    try:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, max_redirects=5
        ) as client:
            resp = await client.get(url, headers={"User-Agent": "RAG-App/1.0"})
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise UrlFetchError(f"Récupération impossible : {exc}") from exc

    if len(resp.content) > MAX_URL_BYTES:
        raise UnreadableDocumentError("Page trop volumineuse (> 5 Mo).")

    title, text = html_to_text(resp.text)
    if not text.strip():
        raise UnreadableDocumentError("Aucun texte extractible de cette page.")

    parsed = urlparse(url)
    filename = (title or f"{parsed.netloc}{parsed.path}").strip()[:255] or parsed.netloc
    document = await ingestor.ingest(
        filename=filename,
        content_type="text/html",
        size_bytes=len(resp.content),
        text=text,
    )
    return DocumentResponse(**document)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: DbDep) -> list[DocumentResponse]:
    return [DocumentResponse(**d) for d in await db.list_documents()]


@router.delete("/{doc_id}")
async def delete_document(doc_id: int, db: DbDep) -> dict:
    if not await db.get_document(doc_id):
        raise NotFoundError("Document not found.")
    deleted = await db.delete_document(doc_id)
    return {"status": "deleted", "chunks_deleted": deleted}
