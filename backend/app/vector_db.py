"""Async database layer (asyncpg + pgvector).

Implements the persistence the CLAUDE.md spec left as TODOs: documents, chunks
(with vector embeddings), conversations and messages, plus cosine-similarity
retrieval over pgvector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import get_settings

logger = logging.getLogger(__name__)

# init_db.sql lives at the backend root (one level above this app/ package).
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "init_db.sql"


class Database:
    """Owns the asyncpg connection pool and all SQL access."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pool: Optional[asyncpg.Pool] = None

    # ---------- Lifecycle ----------
    async def connect(self) -> None:
        """Ensure the schema exists, then create the pgvector-aware pool."""
        # Bootstrap the schema first: register_vector (run on every pooled
        # connection) needs the `vector` type to already exist, so the
        # extension must be created before the pool is built. init_db.sql is
        # fully idempotent (CREATE ... IF NOT EXISTS), so this is safe to run
        # on every startup and removes the need to apply it manually in prod.
        await self._ensure_schema()

        self.pool = await asyncpg.create_pool(
            self.settings.database_url,
            min_size=self.settings.db_pool_min_size,
            max_size=self.settings.db_pool_max_size,
            init=self._init_connection,
        )
        logger.info("Database pool created.")

    async def _ensure_schema(self) -> None:
        """Apply init_db.sql once via a plain (non-pgvector) connection."""
        if not SCHEMA_PATH.exists():
            logger.warning("Schema file not found at %s; skipping bootstrap.", SCHEMA_PATH)
            return
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn = await asyncpg.connect(self.settings.database_url)
        try:
            await conn.execute(sql)
            logger.info("Schema ensured (init_db.sql applied).")
        finally:
            await conn.close()

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        # Lets us pass/receive Python lists as `vector` columns directly.
        await register_vector(conn)

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
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
                filename,
                content_type,
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

    async def get_document(self, document_id: int) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", document_id
            )
            return dict(row) if row else None

    async def list_documents(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            )
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
    async def create_conversation(self, title: Optional[str] = None) -> UUID:
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
        retrieved_chunk_ids: Optional[list[int]] = None,
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

    async def get_conversation(self, conversation_id: UUID) -> Optional[dict]:
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


# Module-level singleton.
database = Database()
