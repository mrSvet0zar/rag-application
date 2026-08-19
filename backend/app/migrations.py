"""Running Alembic migrations from the application.

The app applies `alembic upgrade head` at startup rather than relying on a
separate deploy step: the target (Railway) gives no reliable release hook, and
a service that boots against a stale schema fails in far more confusing ways
than one that migrates itself.

Two safeguards make that acceptable:

* A PostgreSQL advisory lock serialises concurrent instances, so rolling
  restarts or multiple replicas cannot run migrations against each other.
* Alembic runs each migration in a transaction, so a failure leaves the schema
  untouched and the version table unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI = BACKEND_ROOT / "alembic.ini"

# Arbitrary but fixed application-wide key; any instance migrating this
# database takes the same lock. (Chosen once, must never change.)
_MIGRATION_LOCK_KEY = 0x5241_4721  # "RAG!"


def _alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    # alembic.ini's relative script_location resolves against the process CWD,
    # which is not guaranteed to be the backend directory (uvicorn may be
    # started from anywhere, and the container sets its own WORKDIR).
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    # env.py reads this in preference to the ambient settings, so callers
    # migrate the database they asked for and not the one in .env.
    config.attributes["database_url"] = database_url
    return config


def _upgrade_to_head(database_url: str) -> None:
    """Blocking Alembic call — must not run on the event loop."""
    command.upgrade(_alembic_config(database_url), "head")


async def run_migrations(database_url: str) -> None:
    """Bring the database up to the latest revision.

    Holds an advisory lock on a dedicated connection for the duration, so a
    second instance waits here instead of racing on DDL.
    """
    lock_conn = await asyncpg.connect(database_url)
    try:
        await lock_conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_KEY)
        # Alembic is synchronous and spins up its own event loop (env.py runs
        # asyncio.run), so it needs a thread without a running loop.
        await asyncio.to_thread(_upgrade_to_head, database_url)
        logger.info("Migrations applied (alembic upgrade head).")
    finally:
        # Releasing is best-effort: closing the connection drops the lock too.
        try:
            await lock_conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_KEY)
        finally:
            await lock_conn.close()
