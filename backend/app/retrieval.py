"""Retrieval: candidate generation (dense and/or lexical), then reranking."""

from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.fusion import fuse_rows
from app.protocols import Embedder, Reranker, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Finds the chunks most relevant to a question.

    Up to three stages, each narrowing what the next has to look at:

    1. **Candidate generation.** Vector search always runs; with hybrid
       retrieval a lexical search runs alongside it and the two rankings are
       merged with Reciprocal Rank Fusion. The two are complementary — the
       embedding understands paraphrase, the lexical index catches exact terms
       the embedding is blind to.
    2. **Reranking.** A cross-encoder re-scores the shortlist. Accurate but
       slow, so it never sees more than the candidate pool.
    3. **Truncation** to the k the caller asked for.
    """

    def __init__(
        self,
        db: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None,
        settings: Settings,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

    async def retrieve(self, question: str, k: int) -> list[dict]:
        """Return at most `k` chunks, most relevant first."""
        candidates = await self._candidates(question, k)
        if not candidates:
            return []
        if self._reranker is None:
            return candidates[:k]
        return await asyncio.to_thread(self._reranker.rerank, question, candidates, k)

    async def _candidates(self, question: str, k: int) -> list[dict]:
        """Build the shortlist the reranker (or the caller) will see.

        When something downstream will re-score the shortlist, we deliberately
        fetch a broad pool with **no** cosine floor: the cross-encoder, not the
        bi-encoder, should decide relevance, and applying the floor here would
        hide good chunks from it (a chunk at 0.19 cosine has been observed
        reranking to #1). Without reranking we fetch exactly `k` above
        `min_relevance_score`, since nothing later can rescue a bad ordering.
        """
        widening = self._reranker is not None or self._settings.hybrid_enabled
        pool = self._settings.rerank_candidates if widening else k
        floor = 0.0 if widening else self._settings.min_relevance_score

        # Embedding is CPU-bound -> keep it off the event loop.
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, question)
        dense = await self._db.search(query_embedding, top_k=pool, min_score=floor)

        if not self._settings.hybrid_enabled:
            return dense

        lexical = await self._db.search_lexical(
            question, top_k=self._settings.lexical_candidates
        )
        if not lexical:
            # Nothing matched lexically (a query of stopwords, an empty index):
            # fusing a single list would just reorder it for no reason.
            return dense

        fused = fuse_rows([dense, lexical], k=self._settings.rrf_k)
        # Cap on what the cross-encoder will score, not on the pool size of a
        # single retriever: trimming the fused list back to `pool` would throw
        # away most of the lexical contribution before it is ever judged.
        return fused[: self._settings.rerank_max_candidates]
