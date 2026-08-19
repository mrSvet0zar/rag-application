"""Retrieval metrics.

Deliberately hand-written rather than pulled from a library: they are a few
lines each, they are the thing being reasoned about, and implementing them
makes their assumptions explicit. All of them take a *ranked* list of retrieved
ids and the set of ids that are actually relevant, so they are pure functions —
no database, no model, no network.

Relevance is binary here (a chunk either answers the question or it does not),
which is what the golden set expresses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 if any relevant chunk made the top k, else 0.0.

    The bluntest measure, and the one closest to what RAG actually needs: the
    generator only requires *one* good passage to answer.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the relevant chunks that appear in the top k."""
    if not relevant:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / len(relevant)


def precision_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top k that is relevant.

    Divided by k rather than by the number retrieved: returning three good
    chunks out of five asked for is not the same as returning three out of
    three, and the context window pays for the difference.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[int], relevant: set[int]) -> float:
    """1 / rank of the first relevant chunk (0.0 if none).

    Averaged over questions this is MRR. It rewards putting the good passage
    first, which matters because the LLM reads the top of the context best.
    """
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain over the top k.

    Unlike recall it is rank-aware, and unlike MRR it accounts for *every*
    relevant chunk found, discounted by how far down it sits. The ideal ranking
    used for normalisation is all relevant chunks packed at the top.
    """
    if not relevant:
        return 0.0
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0
