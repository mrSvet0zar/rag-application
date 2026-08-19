"""Liveness and readiness probes.

Two endpoints because they answer different questions, and conflating them is
how a deployment ends up green while the service cannot serve anything:

* **liveness** — is the process alive? No I/O, so a slow database never causes
  the platform to kill an otherwise healthy container.
* **readiness** — can it actually serve? Checks the database, and answers 503
  when it cannot, so traffic is withheld instead of failing in the user's face.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status

from app.deps import DbDep, SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: SettingsDep) -> dict:
    """Liveness: deliberately does no I/O."""
    return {"status": "ok", "demo_mode": settings.demo_mode}


@router.get("/ready")
async def ready(db: DbDep, response: Response) -> dict:
    """Readiness: a trivial round-trip proves the pool really works.

    `SELECT 1` rather than an inspection query — it is cheap enough to be probed
    every few seconds, and it exercises the whole path (pool, connection,
    server) which is what actually breaks.
    """
    try:
        async with db.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as exc:  # any failure at all means "not ready"
        logger.warning("Readiness check failed", extra={"error": str(exc)})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "database": "unreachable"}

    return {"status": "ready", "database": "ok"}
