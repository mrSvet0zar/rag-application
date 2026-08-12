"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------- Documents ----------
class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    uploaded_at: datetime
    total_chunks: int
    status: str

    model_config = {"from_attributes": True}


# ---------- Chat ----------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[UUID] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    k: int = Field(5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    id: int
    text: str
    document_id: int
    filename: Optional[str] = None
    similarity_score: float  # cosine similarity from vector search
    rerank_score: Optional[float] = None  # cross-encoder relevance, if reranked


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
    title: Optional[str] = None
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
