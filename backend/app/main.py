"""FastAPI application: document upload, RAG chat, conversations, stats."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.embeddings import embedding_service
from app.rag_pipeline import RAGPipeline
from app.reranker import rerank_service
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    DocumentResponse,
    RetrievedChunk,
    StatsResponse,
)
from app.vector_db import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag")

settings = get_settings()
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: open DB pool and build the RAG pipeline."""
    global rag_pipeline
    await database.connect()
    rag_pipeline = RAGPipeline()
    logger.info("Startup complete (demo_mode=%s).", settings.demo_mode)
    yield
    await database.disconnect()


app = FastAPI(title="RAG Application API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Helpers ============
SUPPORTED_TEXT_TYPES = {".txt", ".md", ".markdown", ".csv", ".json"}


def _extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from an uploaded file (txt/md/pdf)."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # Default: treat as text.
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


async def _retrieve_chunks(question: str, k: int) -> list[dict]:
    """Embed the query, vector-search a candidate pool, then (optionally)
    rerank it down to the top `k` with a cross-encoder.

    When reranking is on we fetch a broad pool (`rerank_candidates`) with **no**
    cosine floor, because the cross-encoder — not the bi-encoder — should decide
    relevance; applying the 0.25 floor here would hide good chunks from it.
    When reranking is off we fetch exactly `k` above `min_relevance_score`.
    """
    query_embedding = await asyncio.to_thread(embedding_service.embed_query, question)
    if settings.rerank_enabled:
        candidates = await database.search(
            query_embedding, top_k=settings.rerank_candidates, min_score=0.0
        )
        if not candidates:
            return []
        return await asyncio.to_thread(rerank_service.rerank, question, candidates, k)
    return await database.search(
        query_embedding, top_k=k, min_score=settings.min_relevance_score
    )


# ============ Health ============
@app.get("/api/health")
async def health():
    return {"status": "ok", "demo_mode": settings.demo_mode}


# ============ Documents ============
@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document: extract text, chunk, embed, and store."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        text = _extract_text(file.filename, content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text in file.")

    chunks = rag_pipeline.split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks.")

    document_id = await database.create_document(
        filename=file.filename,
        content_type=file.content_type or "text/plain",
        file_size_bytes=len(content),
    )

    try:
        # Embedding is CPU-bound -> run off the event loop.
        embeddings = await asyncio.to_thread(embedding_service.embed_documents, chunks)
        chunk_data = [
            (chunk, embedding, {"filename": file.filename})
            for chunk, embedding in zip(chunks, embeddings)
        ]
        stored = await database.store_chunks(document_id, chunk_data)
        await database.finalize_document(document_id, stored, status="completed")
    except Exception as exc:  # noqa: BLE001
        await database.finalize_document(document_id, 0, status="failed")
        logger.exception("Failed to process document %s", document_id)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    doc = await database.get_document(document_id)
    return DocumentResponse(**doc)


@app.get("/api/documents", response_model=list[DocumentResponse])
async def list_documents():
    return [DocumentResponse(**d) for d in await database.list_documents()]


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    doc = await database.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    deleted = await database.delete_document(doc_id)
    return {"status": "deleted", "chunks_deleted": deleted}


# ============ Chat ============
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using retrieval-augmented generation."""
    start = time.time()

    # 1. Retrieve relevant chunks (vector search + optional cross-encoder rerank).
    chunks = await _retrieve_chunks(request.question, request.k)

    # 2. Generate the answer (Claude, or demo fallback).
    answer, tokens = await rag_pipeline.generate_response(request.question, chunks)

    # 3. Persist the conversation and both messages.
    conversation_id = request.conversation_id
    if conversation_id is None or not await database.conversation_exists(conversation_id):
        conversation_id = await database.create_conversation()

    await database.add_message(conversation_id, "user", request.question)
    chunk_ids = [c["id"] for c in chunks]
    message_id = await database.add_message(
        conversation_id, "assistant", answer, retrieved_chunk_ids=chunk_ids, tokens_used=tokens
    )

    processing_ms = (time.time() - start) * 1000
    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        message_id=message_id,
        retrieved_chunks=[
            RetrievedChunk(
                id=c["id"],
                text=c["text"],
                document_id=c["document_id"],
                filename=c.get("filename"),
                similarity_score=round(float(c["similarity"]), 4),
                rerank_score=(
                    round(float(c["rerank_score"]), 4)
                    if c.get("rerank_score") is not None
                    else None
                ),
            )
            for c in chunks
        ],
        tokens_used=tokens,
        processing_time_ms=round(processing_ms, 2),
    )


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Same as /api/chat but streams the answer token-by-token over SSE.

    Event sequence:
      sources -> {retrieved_chunks: [...]}   (once, before generation)
      token   -> {text: "..."}               (many)
      done    -> {conversation_id, message_id, tokens_used, processing_time_ms}
      error   -> {detail: "..."}             (on failure)
    """

    async def event_generator():
        start = time.time()
        try:
            chunks = await _retrieve_chunks(request.question, request.k)

            sources = [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "document_id": c["document_id"],
                    "filename": c.get("filename"),
                    "similarity_score": round(float(c["similarity"]), 4),
                    "rerank_score": (
                        round(float(c["rerank_score"]), 4)
                        if c.get("rerank_score") is not None
                        else None
                    ),
                }
                for c in chunks
            ]
            yield _sse("sources", {"retrieved_chunks": sources})

            parts: list[str] = []
            tokens = 0
            async for kind, data in rag_pipeline.stream_response(
                request.question, chunks
            ):
                if kind == "token":
                    parts.append(data)
                    yield _sse("token", {"text": data})
                elif kind == "usage":
                    tokens = data

            answer = "".join(parts)

            conversation_id = request.conversation_id
            if conversation_id is None or not await database.conversation_exists(
                conversation_id
            ):
                conversation_id = await database.create_conversation()
            await database.add_message(conversation_id, "user", request.question)
            message_id = await database.add_message(
                conversation_id,
                "assistant",
                answer,
                retrieved_chunk_ids=[c["id"] for c in chunks],
                tokens_used=tokens,
            )

            yield _sse(
                "done",
                {
                    "conversation_id": str(conversation_id),
                    "message_id": str(message_id),
                    "tokens_used": tokens,
                    "processing_time_ms": round((time.time() - start) * 1000, 2),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/railway)
        },
    )


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: UUID):
    convo = await database.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse(**convo)


# ============ Stats ============
@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    stats = await database.get_stats()
    return StatsResponse(
        **stats,
        demo_mode=settings.demo_mode,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        rerank_enabled=settings.rerank_enabled,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
