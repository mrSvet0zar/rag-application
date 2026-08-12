-- ============================================================
--  RAG Application - Database schema (PostgreSQL + pgvector)
-- ============================================================
--  Embeddings use a local multilingual sentence-transformers
--  model -> 384 dimensions. Change vector(384) if you swap the
--  model (keep it in sync with PGVECTOR_DIMENSION in .env).
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()

-- ---------- Documents ----------
CREATE TABLE IF NOT EXISTS documents (
  id              SERIAL PRIMARY KEY,
  filename        VARCHAR(255) NOT NULL,
  content_type    VARCHAR(50),
  file_size_bytes BIGINT,
  uploaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  total_chunks    INT DEFAULT 0,
  status          VARCHAR(20) DEFAULT 'processing'
);

-- ---------- Chunks (with embeddings) ----------
CREATE TABLE IF NOT EXISTS chunks (
  id           SERIAL PRIMARY KEY,
  document_id  INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index  INT NOT NULL,
  text         TEXT NOT NULL,
  embedding    vector(384),
  metadata     JSONB,
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT unique_chunk UNIQUE(document_id, chunk_index)
);

-- ---------- Conversations ----------
CREATE TABLE IF NOT EXISTS conversations (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_message_at TIMESTAMP,
  total_messages  INT DEFAULT 0,
  title           VARCHAR(255)
);

-- ---------- Messages ----------
CREATE TABLE IF NOT EXISTS messages (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id     UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role                VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
  content             TEXT NOT NULL,
  retrieved_chunk_ids INT[] DEFAULT ARRAY[]::INT[],
  tokens_used         INT,
  created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------- Indices ----------
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
-- HNSW rather than IVFFlat: IVFFlat must be trained on existing data and, with
-- a large `lists` value on a small/growing table, under-probes and returns no
-- rows. HNSW builds incrementally and gives good recall regardless of size.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
