"""Two-stage retrieval: vector search, then optional cross-encoder reranking."""

from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.protocols import Embedder, Reranker, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Finds the chunks most relevant to a question.

    Stage 1 (bi-encoder) is cheap and runs over the whole corpus; stage 2
    (cross-encoder) is accurate but slow, so it only re-scores the shortlist.
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
        """Return at most `k` chunks, most relevant first.

        With reranking on we deliberately fetch a broad pool with **no** cosine
        floor: the cross-encoder, not the bi-encoder, should decide relevance —
        applying the floor here would hide good chunks from it (a chunk at 0.19
        cosine has been observed reranking to #1). Without reranking we fetch
        exactly `k` above `min_relevance_score`.
        """
        # Embedding is CPU-bound -> keep it off the event loop.
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, question)

        if self._reranker is None:
            return await self._db.search(
                query_embedding,
                top_k=k,
                min_score=self._settings.min_relevance_score,
            )

        candidates = await self._db.search(
            query_embedding, top_k=self._settings.rerank_candidates, min_score=0.0
        )
        if not candidates:
            return []
        return await asyncio.to_thread(self._reranker.rerank, question, candidates, k)
