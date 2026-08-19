"""Migrations must build the schema from nothing, and be safe to re-run.

Every other integration test starts from an already-migrated database, so it
would never notice a migration that only works on a database that already has
the tables. These run against a throwaway database created per test.
"""

from __future__ import annotations

import asyncpg
import pytest

from app.config import Settings
from app.migrations import run_migrations
from tests.integration.conftest import _maintenance_url

EXPECTED_TABLES = {"documents", "chunks", "conversations", "messages"}


@pytest.fixture
async def blank_database(test_settings: Settings):
    """URL of a freshly created, completely empty database, dropped after."""
    name = "rag_migration_probe"
    maintenance = _maintenance_url(test_settings.database_url)

    conn = await asyncpg.connect(maintenance)
    try:
        # Identifier is a literal here, so interpolation is safe.
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()

    url = _maintenance_url(test_settings.database_url).replace("/postgres", f"/{name}")
    try:
        yield url
    finally:
        conn = await asyncpg.connect(maintenance)
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await conn.close()


async def _tables(url: str) -> set[str]:
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        return {r["tablename"] for r in rows}
    finally:
        await conn.close()


async def _scalar(url: str, sql: str):
    conn = await asyncpg.connect(url)
    try:
        return await conn.fetchval(sql)
    finally:
        await conn.close()


async def test_migrates_an_empty_database_to_the_full_schema(blank_database: str):
    assert await _tables(blank_database) == set()

    await run_migrations(blank_database)

    assert await _tables(blank_database) >= EXPECTED_TABLES
    assert (
        await _scalar(blank_database, "SELECT version_num FROM alembic_version")
        == "0001_baseline"
    )


async def test_creates_the_pgvector_column_and_hnsw_index(blank_database: str):
    """The embedding column and its index are what retrieval depends on."""
    await run_migrations(blank_database)

    column_type = await _scalar(
        blank_database,
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'",
    )
    assert column_type == "vector(384)"

    index = await _scalar(
        blank_database,
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_chunks_embedding'",
    )
    assert "hnsw" in index.lower()


async def test_running_migrations_twice_is_a_no_op(blank_database: str):
    """Startup runs migrations on every boot, so re-running must be harmless."""
    await run_migrations(blank_database)
    await run_migrations(blank_database)

    assert await _scalar(blank_database, "SELECT count(*) FROM alembic_version") == 1


async def test_migrates_a_database_that_already_has_the_legacy_schema(
    blank_database: str,
):
    """Adopting Alembic on the pre-existing production database must not fail.

    Before Alembic, the schema was applied by an idempotent init_db.sql. The
    baseline revision has to run cleanly — and without touching data — against
    a database already in that state.
    """
    conn = await asyncpg.connect(blank_database)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            """CREATE TABLE documents (
                 id SERIAL PRIMARY KEY,
                 filename VARCHAR(255) NOT NULL,
                 content_type VARCHAR(50),
                 file_size_bytes BIGINT,
                 uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 total_chunks INT DEFAULT 0,
                 status VARCHAR(20) DEFAULT 'processing')"""
        )
        await conn.execute("INSERT INTO documents (filename) VALUES ('pre-existing.md')")
    finally:
        await conn.close()

    await run_migrations(blank_database)

    assert await _tables(blank_database) >= EXPECTED_TABLES
    # The row that was there before the migration is still there.
    assert (
        await _scalar(blank_database, "SELECT filename FROM documents")
        == "pre-existing.md"
    )
