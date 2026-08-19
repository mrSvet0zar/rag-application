"""Add a full-text search vector to chunks, for hybrid retrieval.

Revision ID: 0002_lexical
Revises: 0001_baseline
"""

from __future__ import annotations

from alembic import op

revision: str = "0002_lexical"
down_revision: str | None = "0001_baseline"
branch_labels = None
depends_on = None

# Vector search alone is blind to exact terms — acronyms, proper nouns, English
# titles inside French prose. A lexical index covers precisely that gap, and
# PostgreSQL provides one natively, so hybrid retrieval needs no second service.
#
# The column is GENERATED ... STORED rather than maintained by the application:
# it can never drift from `text`, and it is backfilled for existing rows when
# this migration runs.
#
# `to_tsvector('french', ...)` stems French words but does *not* strip accents,
# so a query typed without accents will not match accented text. Making it
# accent-insensitive means wrapping `unaccent` in a function marked IMMUTABLE —
# which it is not, and lying about that can silently corrupt the index if the
# dictionary ever changes. The safe, honest option is chosen here.
UPGRADE: tuple[str, ...] = (
    """
    ALTER TABLE chunks
      ADD COLUMN search_vector tsvector
      GENERATED ALWAYS AS (to_tsvector('french', text)) STORED
    """,
    "CREATE INDEX idx_chunks_search_vector ON chunks USING gin (search_vector)",
)

DOWNGRADE: tuple[str, ...] = (
    "DROP INDEX IF EXISTS idx_chunks_search_vector",
    "ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector",
)


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
