"""Aggregate statistics endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import DbDep, SettingsDep
from app.schemas import StatsResponse

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: DbDep, settings: SettingsDep) -> StatsResponse:
    stats = await db.get_stats()
    return StatsResponse(
        **stats,
        demo_mode=settings.demo_mode,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        rerank_enabled=settings.rerank_enabled,
    )
