"""Pydantic request/response models for the API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# ---------- Documents ----------
class UrlImportRequest(BaseModel):
    url: HttpUrl


class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: str | None = None
    file_size_bytes: int | None = None
    uploaded_at: datetime
    total_chunks: int
    status: str

    model_config = {"from_attributes": True}


# ---------- Chat ----------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    k: int = Field(5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    id: int
    text: str
    document_id: int
    filename: str | None = None
    # Each score is absent when the stage that produces it did not see this
    # chunk: a passage found only by the lexical search has no cosine
    # similarity, and vice versa. Reporting 0.0 instead of null would claim a
    # measurement that was never made.
    similarity_score: float | None = None  # cosine similarity, if vector-matched
    lexical_score: float | None = None  # full-text rank, if lexically matched
    rerank_score: float | None = None  # cross-encoder relevance, if reranked


class ChatResponse(BaseModel):
    response: str
    conversation_id: UUID
    message_id: UUID
    retrieved_chunks: list[RetrievedChunk]
    tokens_used: int
    processing_time_ms: float


# ---------- Conversations ----------
class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: UUID
    created_at: datetime
    title: str | None = None
    messages: list[MessageResponse]


# ---------- Stats ----------
class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_conversations: int
    total_messages: int
    demo_mode: bool
    chat_model: str
    embedding_model: str
    rerank_enabled: bool = False
