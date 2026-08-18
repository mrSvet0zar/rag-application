"""Conversation history endpoint."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from app.deps import DbDep
from app.errors import NotFoundError
from app.schemas import ConversationResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: UUID, db: DbDep) -> ConversationResponse:
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise NotFoundError("Conversation not found.")
    return ConversationResponse(**convo)
