"""Alembic environment.

Two deliberate choices:

* The URL comes from the application `Settings`, not from alembic.ini, so
  connection details (and credentials) live in exactly one place.
* There is no `target_metadata` and autogenerate is unused: this project talks
  to PostgreSQL through asyncpg with hand-written SQL, not an ORM, so there are
  no SQLAlchemy models to diff against. Migrations are explicit SQL, which also
  keeps pgvector-specific DDL (`vector(384)`, HNSW) fully under our control.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from alembic import context
from app.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _database_url() -> str:
    """The database to migrate, adapted to SQLAlchemy's asyncpg dialect.

    Priority matters: when the application runs migrations itself it passes the
    URL through `config.attributes`, which must win over the ambient settings —
    otherwise the test suite (pointed at TEST_DATABASE_URL) would silently
    migrate whatever database the local .env happens to name. Falling back to
    Settings keeps the plain `alembic` CLI working.

    Settings hold a libpq-style `postgresql://` URL (what asyncpg.connect and
    psql expect); SQLAlchemy needs the driver spelled out.
    """
    url = config.attributes.get("database_url") or get_settings().database_url
    if url.startswith("postgresql+"):
        return url
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (`alembic upgrade --sql`)."""
    context.configure(
        url=_database_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database."""
    config.set_main_option("sqlalchemy.url", _database_url())
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # Migrations are a short-lived, one-shot task: pooling would only keep
        # idle connections around after we're done.
        poolclass=NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
