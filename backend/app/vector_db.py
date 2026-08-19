"""Async database layer (asyncpg + pgvector).

Implements the persistence the CLAUDE.md spec left as TODOs: documents, chunks
(with vector embeddings), conversations and messages, plus cosine-similarity
retrieval over pgvector.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings
from app.migrations import run_migrations

logger = logging.getLogger(__name__)


class Database:
    """Owns the asyncpg connection pool and all SQL access."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        """The connection pool, or a clear error if `connect()` hasn't run.

        Without this, every query site would need a None check to satisfy the
        type checker, and a misuse would surface as an opaque
        `AttributeError: 'NoneType' object has no attribute 'acquire'`.
        """
        if self._pool is None:
            raise RuntimeError(
                "Database.connect() must be awaited before running queries."
            )
        return self._pool

    # ---------- Lifecycle ----------
    async def connect(self, max_wait_seconds: float = 90.0) -> None:
        """Connect to the DB, retrying while the network/DB comes up.

        Railway's private network (`*.railway.internal`) takes a few seconds to
        be established after a service starts, so an immediate connection at
        boot can fail DNS resolution. We retry with exponential backoff until
        the DB is reachable (or the budget is exhausted), which also covers the
        DB simply not being ready yet.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait_seconds
        attempt = 0
        delay = 1.0
        while True:
            attempt += 1
            try:
                await self._bootstrap()
                logger.info("Database ready (after %d attempt(s)).", attempt)
                return
            except (OSError, asyncpg.PostgresError) as exc:
                if loop.time() >= deadline:
                    logger.error(
                        "Could not connect to the database after %d attempts / "
                        "%.0fs. Last error: %s",
                        attempt,
                        max_wait_seconds,
                        exc,
                    )
                    raise
                logger.warning(
                    "DB not reachable yet (attempt %d): %s. Retrying in %.0fs...",
                    attempt,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)

    async def _bootstrap(self) -> None:
        """One connection attempt: migrate the schema, then build the pool.

        Order matters: `register_vector` runs on every pooled connection and
        needs the `vector` type to already exist, so migrations (which create
        the extension) must complete before the pool is created.
        """
        await run_migrations(self.settings.database_url)
        self._pool = await asyncpg.create_pool(
            self.settings.database_url,
            min_size=self.settings.db_pool_min_size,
            max_size=self.settings.db_pool_max_size,
            init=self._init_connection,
        )
        logger.info("Database pool created.")

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        # Lets us pass/receive Python lists as `vector` columns directly.
        await register_vector(conn)

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed.")

    # ---------- Documents ----------
    async def create_document(
        self, filename: str, content_type: str, file_size_bytes: int
    ) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """INSERT INTO documents (filename, content_type, file_size_bytes, status)
                   VALUES ($1, $2, $3, 'processing')
                   RETURNING id""",
                filename[:255],
                (content_type or "")[
                    :50
                ],  # column is VARCHAR(50); MIME types can be longer
                file_size_bytes,
            )

    async def finalize_document(
        self, document_id: int, total_chunks: int, status: str = "completed"
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET total_chunks = $2, status = $3 WHERE id = $1",
                document_id,
                total_chunks,
                status,
            )

    async def get_document(self, document_id: int) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", document_id
            )
            return dict(row) if row else None

    async def list_documents(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM documents ORDER BY uploaded_at DESC")
            return [dict(r) for r in rows]

    async def delete_document(self, document_id: int) -> int:
        """Delete a document; chunks cascade. Returns chunks removed."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1", document_id
            )
            await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
            return count or 0

    # ---------- Chunks ----------
    async def store_chunks(
        self, document_id: int, chunks: list[tuple[str, list[float], dict]]
    ) -> int:
        """Bulk-insert (text, embedding, metadata) chunks for a document."""
        records = [
            (document_id, idx, text, embedding, json.dumps(metadata))
            for idx, (text, embedding, metadata) in enumerate(chunks)
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO chunks (document_id, chunk_index, text, embedding, metadata)
                   VALUES ($1, $2, $3, $4, $5)""",
                records,
            )
        return len(records)

    # Must match the configuration used by the generated `search_vector`
    # column (see migration 0002): a query parsed with a different dictionary
    # would produce lexemes the index never stored.
    FTS_CONFIG = "french"

    async def search_lexical(self, query: str, top_k: int = 20) -> list[dict]:
        """Full-text search over chunks, best first.

        The query text is reduced to its lexemes and combined with OR rather
        than AND. `websearch_to_tsquery` and `plainto_tsquery` both AND their
        terms, which for a natural-language question means every content word
        must appear in the same chunk — in practice that returns nothing at all.
        ORing keeps it a ranking problem instead of a filter.

        Note this is `ts_rank_cd`, not BM25: PostgreSQL's ranking has no inverse
        document frequency, so a rare term is not rewarded over a common one.
        It is still enough to surface exact matches that the embedding misses.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""WITH parsed AS (
                       SELECT to_tsquery('{self.FTS_CONFIG}', (
                           SELECT string_agg(quote_literal(lexeme), ' | ')
                           FROM unnest(to_tsvector('{self.FTS_CONFIG}', $1))
                       )) AS tsq
                   )
                   SELECT c.id,
                          c.document_id,
                          c.text,
                          d.filename,
                          ts_rank_cd(c.search_vector, parsed.tsq) AS lexical_score
                   FROM chunks c
                   JOIN documents d ON d.id = c.document_id,
                        parsed
                   WHERE parsed.tsq IS NOT NULL
                     AND c.search_vector @@ parsed.tsq
                   ORDER BY lexical_score DESC
                   LIMIT $2""",
                query,
                top_k,
            )
            return [dict(r) for r in rows]

    async def search(
        self, query_embedding: list[float], top_k: int = 5, min_score: float = 0.25
    ) -> list[dict]:
        """Cosine-similarity search over chunk embeddings.

        `1 - (embedding <=> query)` converts pgvector cosine distance to a
        similarity in [-1, 1] (1 = identical).
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT c.id,
                          c.document_id,
                          c.text,
                          d.filename,
                          1 - (c.embedding <=> $1) AS similarity
                   FROM chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE 1 - (c.embedding <=> $1) > $2
                   ORDER BY c.embedding <=> $1
                   LIMIT $3""",
                query_embedding,
                min_score,
                top_k,
            )
            return [dict(r) for r in rows]

    # ---------- Conversations & messages ----------
    async def create_conversation(self, title: str | None = None) -> UUID:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """INSERT INTO conversations (title, last_message_at)
                   VALUES ($1, CURRENT_TIMESTAMP)
                   RETURNING id""",
                title,
            )

    async def conversation_exists(self, conversation_id: UUID) -> bool:
        async with self.pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT 1 FROM conversations WHERE id = $1", conversation_id
                )
            )

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        retrieved_chunk_ids: list[int] | None = None,
        tokens_used: int = 0,
    ) -> UUID:
        async with self.pool.acquire() as conn:
            message_id = await conn.fetchval(
                """INSERT INTO messages
                     (conversation_id, role, content, retrieved_chunk_ids, tokens_used)
                   VALUES ($1, $2, $3, $4, $5)
                   RETURNING id""",
                conversation_id,
                role,
                content,
                retrieved_chunk_ids or [],
                tokens_used,
            )
            await conn.execute(
                """UPDATE conversations
                   SET total_messages = total_messages + 1,
                       last_message_at = CURRENT_TIMESTAMP,
                       title = COALESCE(title, LEFT($2, 60))
                   WHERE id = $1""",
                conversation_id,
                content if role == "user" else None,
            )
            return message_id

    async def get_conversation(self, conversation_id: UUID) -> dict | None:
        async with self.pool.acquire() as conn:
            convo = await conn.fetchrow(
                "SELECT * FROM conversations WHERE id = $1", conversation_id
            )
            if not convo:
                return None
            messages = await conn.fetch(
                """SELECT id, role, content, created_at
                   FROM messages WHERE conversation_id = $1
                   ORDER BY created_at ASC""",
                conversation_id,
            )
            return {**dict(convo), "messages": [dict(m) for m in messages]}

    # ---------- Stats ----------
    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            return {
                "total_documents": await conn.fetchval("SELECT COUNT(*) FROM documents"),
                "total_chunks": await conn.fetchval("SELECT COUNT(*) FROM chunks"),
                "total_conversations": await conn.fetchval(
                    "SELECT COUNT(*) FROM conversations"
                ),
                "total_messages": await conn.fetchval("SELECT COUNT(*) FROM messages"),
            }
