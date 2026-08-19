"""Baseline schema: documents, chunks (pgvector), conversations, messages.

Revision ID: 0001_baseline
Revises:
"""

from __future__ import annotations

from alembic import op

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels = None
depends_on = None

# This baseline reproduces the schema as it existed *before* Alembic was
# introduced, when an idempotent init_db.sql was applied at startup.
#
# It therefore keeps `IF NOT EXISTS` so that `alembic upgrade head` is safe
# against a database already carrying that schema (the live Railway one) —
# adopting Alembic on an existing database without a manual `alembic stamp`.
# Later migrations are ordinary, non-idempotent DDL: from here on Alembic owns
# the schema and the version table records exactly what has been applied.
#
# Statements are listed one per entry rather than as a single script because
# the asyncpg driver prepares every statement it sends, and a prepared
# statement cannot contain multiple commands.

UPGRADE: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    'CREATE EXTENSION IF NOT EXISTS "pgcrypto"',  # gen_random_uuid()
    """
    CREATE TABLE IF NOT EXISTS documents (
      id              SERIAL PRIMARY KEY,
      filename        VARCHAR(255) NOT NULL,
      content_type    VARCHAR(50),
      file_size_bytes BIGINT,
      uploaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      total_chunks    INT DEFAULT 0,
      status          VARCHAR(20) DEFAULT 'processing'
    )
    """,
    # embedding is 384-dimensional: the local multilingual MiniLM model.
    # Swapping the model means a migration altering this column plus a reindex.
    """
    CREATE TABLE IF NOT EXISTS chunks (
      id           SERIAL PRIMARY KEY,
      document_id  INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      chunk_index  INT NOT NULL,
      text         TEXT NOT NULL,
      embedding    vector(384),
      metadata     JSONB,
      created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      CONSTRAINT unique_chunk UNIQUE(document_id, chunk_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
      id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      last_message_at TIMESTAMP,
      total_messages  INT DEFAULT 0,
      title           VARCHAR(255)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
      id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      conversation_id     UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role                VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
      content             TEXT NOT NULL,
      retrieved_chunk_ids INT[] DEFAULT ARRAY[]::INT[],
      tokens_used         INT,
      created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)",
    # HNSW rather than IVFFlat: IVFFlat must be trained on existing data and,
    # with a large `lists` value on a small/growing table, under-probes and
    # returns no rows. HNSW builds incrementally and keeps good recall at any
    # size.
    """
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
      ON chunks USING hnsw (embedding vector_cosine_ops)
    """,
    "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
)

# Reverse dependency order; extensions are left in place because other
# databases on the same server may rely on them.
DOWNGRADE: tuple[str, ...] = (
    "DROP TABLE IF EXISTS messages",
    "DROP TABLE IF EXISTS conversations",
    "DROP TABLE IF EXISTS chunks",
    "DROP TABLE IF EXISTS documents",
)


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
