"""Document endpoints: upload, URL import, list, delete."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status

from app.deps import DbDep, IngestorDep, SettingsDep
from app.errors import (
    NotFoundError,
    UnreadableDocumentError,
    UrlFetchError,
    UrlNotAllowedError,
)
from app.ingestion import (
    extract_text,
    html_to_text,
    read_capped,
    validate_public_url,
)
from app.schemas import DocumentResponse, UrlImportRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_document(
    ingestor: IngestorDep,
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentResponse:
    """Accept a document (txt/md/pdf/docx/html) and index it in the background.

    Returns 202 with the row in `processing`: chunking and embedding a large
    file takes longer than a client will hold the connection open, so the work
    is detached and the client polls the document's status. Anything cheap
    enough to fail fast — unreadable file, no extractable text, no chunks — is
    still checked here, while there is someone to tell.
    """
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

    document = await ingestor.create_pending(
        filename=filename,
        content_type=file.content_type or "text/plain",
        size_bytes=len(content),
        text=text,
    )
    background.add_task(ingestor.process, document["id"], filename, text)
    return DocumentResponse(**document)


@router.post(
    "/import-url", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED
)
async def import_url(
    req: UrlImportRequest,
    ingestor: IngestorDep,
    settings: SettingsDep,
    background: BackgroundTasks,
) -> DocumentResponse:
    """Fetch a web page, extract its readable text, and index it."""
    url = str(req.url)

    # SSRF guard (blocks localhost / private / metadata addresses). DNS
    # resolution is blocking, hence the thread.
    try:
        await asyncio.to_thread(validate_public_url, url)
    except ValueError as exc:
        raise UrlNotAllowedError(f"URL refusée : {exc}") from exc

    limit = settings.max_upload_bytes
    try:
        # Streamed, not `client.get`: a plain get buffers the entire body first,
        # so checking its size afterwards would happen only once the memory had
        # already been spent. Here we stop mid-download.
        async with (
            httpx.AsyncClient(
                timeout=20.0, follow_redirects=True, max_redirects=5
            ) as client,
            client.stream("GET", url, headers={"User-Agent": "RAG-App/1.0"}) as resp,
        ):
            resp.raise_for_status()
            raw = await read_capped(resp.aiter_bytes(), limit, "Page")
            encoding = resp.encoding or "utf-8"
    except httpx.HTTPError as exc:
        raise UrlFetchError(f"Récupération impossible : {exc}") from exc

    title, text = html_to_text(raw.decode(encoding, errors="replace"))
    if not text.strip():
        raise UnreadableDocumentError("Aucun texte extractible de cette page.")

    parsed = urlparse(url)
    filename = (title or f"{parsed.netloc}{parsed.path}").strip()[:255] or parsed.netloc
    document = await ingestor.create_pending(
        filename=filename,
        content_type="text/html",
        size_bytes=len(raw),
        text=text,
    )
    background.add_task(ingestor.process, document["id"], filename, text)
    return DocumentResponse(**document)


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: int, db: DbDep) -> DocumentResponse:
    """One document, including its ingestion status — what a client polls."""
    document = await db.get_document(doc_id)
    if not document:
        raise NotFoundError("Document not found.")
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
