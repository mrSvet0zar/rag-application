"""Liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: SettingsDep) -> dict:
    return {"status": "ok", "demo_mode": settings.demo_mode}
