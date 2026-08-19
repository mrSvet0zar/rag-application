"""Chat endpoints: buffered JSON answer and token-by-token SSE stream."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps import DbDep, GeneratorDep, RetrieverDep
from app.observability import timed
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk
from app.vector_db import Database

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _to_source(chunk: dict) -> RetrievedChunk:
    """Map a retrieved chunk row to its API representation."""

    def score(name: str) -> float | None:
        value = chunk.get(name)
        return round(float(value), 4) if value is not None else None

    return RetrievedChunk(
        id=chunk["id"],
        text=chunk["text"],
        document_id=chunk["document_id"],
        filename=chunk.get("filename"),
        similarity_score=score("similarity"),
        lexical_score=score("lexical_score"),
        rerank_score=score("rerank_score"),
    )


async def _persist_exchange(
    db: Database,
    conversation_id: UUID | None,
    question: str,
    answer: str,
    chunks: list[dict],
    tokens: int,
) -> tuple[UUID, UUID]:
    """Store the user question and the assistant answer.

    An unknown or missing conversation id starts a new conversation rather than
    failing, so a client holding a stale id keeps working.
    """
    if conversation_id is None or not await db.conversation_exists(conversation_id):
        conversation_id = await db.create_conversation()

    await db.add_message(conversation_id, "user", question)
    message_id = await db.add_message(
        conversation_id,
        "assistant",
        answer,
        retrieved_chunk_ids=[c["id"] for c in chunks],
        tokens_used=tokens,
    )
    return conversation_id, message_id


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    retriever: RetrieverDep,
    generator: GeneratorDep,
    db: DbDep,
) -> ChatResponse:
    """Answer a question using retrieval-augmented generation."""
    start = time.perf_counter()
    stages: dict[str, float] = {}

    with timed(stages, "retrieval_ms"):
        chunks = await retriever.retrieve(request.question, request.k)
    with timed(stages, "generation_ms"):
        answer, tokens = await generator.generate_response(request.question, chunks)
    with timed(stages, "persist_ms"):
        conversation_id, message_id = await _persist_exchange(
            db, request.conversation_id, request.question, answer, chunks, tokens
        )

    _log_answered(stages, chunks, tokens, streamed=False)

    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        message_id=message_id,
        retrieved_chunks=[_to_source(c) for c in chunks],
        tokens_used=tokens,
        processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
    )


def _log_answered(
    stages: dict[str, float], chunks: list[dict], tokens: int, *, streamed: bool
) -> None:
    """One structured line per answered question.

    The stage breakdown is the point: a two-second question is not actionable,
    but knowing the cross-encoder spent most of it is. Correlated with the rest
    of the request through the id the middleware set.
    """
    logger.info(
        "answered",
        extra={
            "event": "chat.answered",
            "streamed": streamed,
            "sources": len(chunks),
            "tokens_used": tokens,
            "total_ms": round(sum(stages.values()), 2),
            **stages,
        },
    )


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    retriever: RetrieverDep,
    generator: GeneratorDep,
    db: DbDep,
) -> StreamingResponse:
    """Same as /chat but streams the answer over SSE.

    Event sequence:
      sources -> {retrieved_chunks: [...]}   (once, before generation)
      token   -> {text: "..."}               (many)
      done    -> {conversation_id, message_id, tokens_used, processing_time_ms}
      error   -> {detail: "..."}             (on failure, terminal)
    """

    async def event_generator() -> AsyncIterator[str]:
        start = time.perf_counter()
        stages: dict[str, float] = {}
        try:
            with timed(stages, "retrieval_ms"):
                chunks = await retriever.retrieve(request.question, request.k)
            sources = [_to_source(c).model_dump() for c in chunks]
            yield _sse("sources", {"retrieved_chunks": sources})

            parts: list[str] = []
            tokens = 0
            generation_started = time.perf_counter()
            first_token_at: float | None = None
            async for kind, data in generator.stream_response(request.question, chunks):
                if kind == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    parts.append(str(data))
                    yield _sse("token", {"text": data})
                elif kind == "usage":
                    tokens = int(data)  # type: ignore[call-overload]
            stages["generation_ms"] = round(
                (time.perf_counter() - generation_started) * 1000, 2
            )
            if first_token_at is not None:
                # What the user actually perceives as latency when streaming:
                # everything after the first token arrives progressively.
                stages["time_to_first_token_ms"] = round(
                    (first_token_at - start) * 1000, 2
                )

            with timed(stages, "persist_ms"):
                conversation_id, message_id = await _persist_exchange(
                    db,
                    request.conversation_id,
                    request.question,
                    "".join(parts),
                    chunks,
                    tokens,
                )

            _log_answered(stages, chunks, tokens, streamed=True)

            yield _sse(
                "done",
                {
                    "conversation_id": str(conversation_id),
                    "message_id": str(message_id),
                    "tokens_used": tokens,
                    "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
        except Exception as exc:
            # The response has already started, so the only way to tell the
            # client is an SSE error event.
            logger.exception(
                "Streaming chat failed", extra={"event": "chat.failed", **stages}
            )
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
