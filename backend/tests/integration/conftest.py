"""Integration test fixtures: a throwaway PostgreSQL + a wired app.

These tests run against a *real* pgvector database, because the bugs worth
catching here (index behaviour, cosine ordering, cascade deletes, column
widths) are exactly the ones an in-memory fake would hide. The heavyweight
ML models are still replaced by deterministic doubles — see tests/doubles.py.

Point them at a database with TEST_DATABASE_URL; the database is created on
demand and every table is truncated between tests.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from app.chunking import TextChunker
from app.config import Settings
from app.ingestor import DocumentIngestor
from app.main import create_app
from app.retrieval import Retriever
from app.services import Services
from app.vector_db import Database
from tests.doubles import FakeEmbedder, StubGenerator

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5432/rag_test",
)

TABLES = ("messages", "conversations", "chunks", "documents")


def _maintenance_url(url: str) -> str:
    """Same server, but the always-present `postgres` database."""
    return urlunsplit(urlsplit(url)._replace(path="/postgres"))


def _database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/")


async def _create_database_if_missing(url: str) -> None:
    conn = await asyncpg.connect(_maintenance_url(url))
    try:
        name = _database_name(url)
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            # Identifier can't be parameterised; it comes from our own config.
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings pinned for tests: no API key, reranking off unless a test asks."""
    return Settings(
        database_url=TEST_DATABASE_URL,
        anthropic_api_key="",
        rerank_enabled=False,
        db_pool_min_size=1,
        db_pool_max_size=4,
        chunk_size=400,
        chunk_overlap=50,
        min_relevance_score=0.0,
        # Off unless a test opts in, so unrelated tests are never throttled.
        rate_limit_enabled=False,
    )


@pytest.fixture(scope="session", autouse=True)
def _database_available(test_settings: Settings) -> None:
    """Create the test database, or skip the whole suite if there's no server."""
    try:
        asyncio.run(_create_database_if_missing(test_settings.database_url))
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(
            f"No PostgreSQL at {_maintenance_url(test_settings.database_url)} "
            f"({exc}). Start it with `docker compose up -d` or set "
            "TEST_DATABASE_URL.",
            allow_module_level=True,
        )


async def _truncate(db: Database) -> None:
    async with db.pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")


@pytest.fixture
async def db(test_settings: Settings):
    """A connected Database on a clean schema (tables emptied)."""
    database = Database(test_settings)
    await database.connect()  # also runs Alembic migrations
    await _truncate(database)
    try:
        yield database
    finally:
        await database.disconnect()


@dataclass
class Harness:
    """An HTTP client plus handles on the doubles behind it."""

    client: AsyncClient
    services: Services
    embedder: FakeEmbedder
    generator: StubGenerator

    @property
    def db(self) -> Database:
        return self.services.db


@pytest.fixture
async def harness(test_settings: Settings, request: pytest.FixtureRequest):
    """A fully wired app over the test database, with fake models.

    Mark a test with `@pytest.mark.rerank` to wire a reranker in.
    """
    from tests.doubles import ReversingReranker

    reranker = ReversingReranker() if request.node.get_closest_marker("rerank") else None

    # Body-size tests would otherwise have to ship 10 MB of payload.
    body_limit = request.node.get_closest_marker("max_body")
    if body_limit is not None:
        test_settings = test_settings.model_copy(
            update={"max_upload_bytes": body_limit.args[0]}
        )

    # Rate-limit tests would otherwise have to fire the full production quota.
    quota = request.node.get_closest_marker("rate_limit")
    if quota is not None:
        test_settings = test_settings.model_copy(
            update={"rate_limit_enabled": True, "rate_limit_requests": quota.args[0]}
        )

    embedder = FakeEmbedder()
    generator = StubGenerator()
    database = Database(test_settings)
    chunker = TextChunker(test_settings.chunk_size, test_settings.chunk_overlap)

    services = Services(
        settings=test_settings,
        db=database,
        embedder=embedder,
        reranker=reranker,
        chunker=chunker,
        generator=generator,
        retriever=Retriever(database, embedder, reranker, test_settings),
        ingestor=DocumentIngestor(database, embedder, chunker),
    )

    app = create_app(test_settings, services_builder=lambda _: services)

    # ASGITransport does not run startup/shutdown, so drive the lifespan
    # explicitly — that way the tests exercise the real wiring.
    async with app.router.lifespan_context(app):
        await _truncate(database)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield Harness(
                client=client,
                services=services,
                embedder=embedder,
                generator=generator,
            )
