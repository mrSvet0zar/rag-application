"""Reciprocal Rank Fusion.

Merging a vector ranking with a lexical one means combining a cosine similarity
(roughly 0 to 1, dense) with `ts_rank_cd` (unbounded, sparse, scaled by term
density). Those scales are not comparable, and normalising them requires
choosing a weighting that is really a hidden hyperparameter.

RRF sidesteps the problem by discarding the scores and keeping only the *ranks*:
each list contributes ``1 / (k + rank)`` to every item it ranks. An item found
by both retrievers accumulates from both and outranks one found by only one,
which is exactly the behaviour hybrid search is after.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Merge ranked id lists into a single ranking, best first.

    `k` damps the advantage of the very top positions: with k=60 (the value
    from the original paper) rank 1 and rank 2 are close, so agreement between
    retrievers matters more than a narrow win inside one of them.

    Ties break on the id, so the output is deterministic — a run-to-run
    reshuffle would show up as noise in the evaluation.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, item in enumerate(ranking, start=1):
            # A ranking should not repeat an id, but if it does only its best
            # position counts: otherwise a duplicate would quietly inflate the
            # score as if two retrievers had agreed.
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def fuse_rows(
    rankings: Sequence[Sequence[dict]], k: int = 60, key: str = "id"
) -> list[dict]:
    """RRF over rows, merging the fields each retriever contributed.

    A chunk found by both searches keeps its vector `similarity` *and* its
    `lexical_score`, so a caller (or a reader of the API response) can still
    tell where a result came from.
    """
    merged: dict[int, dict] = {}
    for ranking in rankings:
        for row in ranking:
            existing = merged.setdefault(row[key], {})
            existing.update(row)

    order = reciprocal_rank_fusion(
        [[row[key] for row in ranking] for ranking in rankings], k=k
    )
    return [merged[item] for item in order]
