"""Cross-encoder reranking.

A bi-encoder (the embedding model) encodes the query and each chunk
*separately*, so it can only approximate their relevance. A cross-encoder
instead reads the (query, chunk) pair *together* and outputs a direct relevance
score — much more accurate, but too slow to run over the whole corpus. The usual
pattern (and the one used here) is: cheap vector search to get a candidate pool,
then cross-encoder to rerank that small pool.

The model is loaded lazily and scoring runs on CPU, so callers should invoke
`rerank` inside a thread (see main.py).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


class RerankService:
    """Reranks retrieved chunks with a multilingual cross-encoder."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info(
                        "Loading reranker model '%s' ...", self._settings.rerank_model
                    )
                    self._model = CrossEncoder(self._settings.rerank_model)
                    logger.info("Reranker model loaded.")
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        """Return the `top_n` chunks most relevant to `query`, reranked.

        Each returned chunk gets a `rerank_score` in [0, 1] (sigmoid of the
        cross-encoder logit). Input order is otherwise irrelevant.
        """
        if not chunks:
            return []
        model = self._ensure_model()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)
        return self._assemble(
            chunks, scores, top_n, min_score=self._settings.rerank_min_score
        )

    @staticmethod
    def _assemble(
        chunks: list[dict],
        scores: Sequence[float],
        top_n: int,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Attach normalized scores, sort by relevance, drop those below
        `min_score` (but always keep the single best), and keep the top_n.

        Pure/deterministic so it can be unit-tested without loading a model.
        """
        scored = []
        for chunk, raw in zip(chunks, scores, strict=False):
            enriched = dict(chunk)
            enriched["rerank_score"] = _sigmoid(float(raw))
            scored.append(enriched)
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        if min_score > 0:
            relevant = [c for c in scored if c["rerank_score"] >= min_score]
            # Never return nothing when we had candidates: keep the best one.
            scored = relevant or scored[:1]
        return scored[:top_n]


def _sigmoid(x: float) -> float:
    # Guard against overflow for very negative logits.
    if x < 0:
        z = math.exp(x)
        return z / (1.0 + z)
    return 1.0 / (1.0 + math.exp(-x))
