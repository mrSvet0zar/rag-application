"""The metrics decide whether every later change counts as an improvement, so
they are pinned against hand-computed values rather than trusted by eye."""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = [10, 20, 30, 40, 50]
RELEVANT = {20, 50, 99}  # 99 is relevant but never retrieved


def test_hit_rate_is_one_as_soon_as_a_relevant_chunk_is_in_range():
    assert hit_rate_at_k(RANKED, RELEVANT, k=2) == 1.0  # 20 is at rank 2
    assert hit_rate_at_k(RANKED, RELEVANT, k=1) == 0.0  # only 10


def test_recall_counts_relevant_found_over_relevant_total():
    assert recall_at_k(RANKED, RELEVANT, k=5) == pytest.approx(2 / 3)  # 20, 50 of 3
    assert recall_at_k(RANKED, RELEVANT, k=2) == pytest.approx(1 / 3)


def test_precision_divides_by_k_not_by_what_was_returned():
    """Three good chunks out of five costs context that three out of three does not."""
    assert precision_at_k(RANKED, RELEVANT, k=5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, k=2) == pytest.approx(1 / 2)
    # Fewer results than k still divides by k.
    assert precision_at_k([20], RELEVANT, k=5) == pytest.approx(1 / 5)


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)
    assert reciprocal_rank([20, 10], RELEVANT) == 1.0
    assert reciprocal_rank([10, 30, 40], RELEVANT) == 0.0


def test_ndcg_rewards_ranking_relevant_chunks_higher():
    perfect = ndcg_at_k([20, 50, 10], {20, 50}, k=3)
    reversed_order = ndcg_at_k([10, 20, 50], {20, 50}, k=3)

    assert perfect == pytest.approx(1.0)
    assert reversed_order < perfect


def test_ndcg_matches_the_hand_computed_value():
    # One relevant chunk at rank 2: gain = 1/log2(3), ideal = 1/log2(2) = 1.
    assert ndcg_at_k([10, 20, 30], {20}, k=3) == pytest.approx(1 / math.log2(3))


@pytest.mark.parametrize(
    "metric", [hit_rate_at_k, recall_at_k, precision_at_k, ndcg_at_k]
)
def test_metrics_are_zero_when_nothing_was_retrieved(metric):
    assert metric([], RELEVANT, 5) == 0.0


@pytest.mark.parametrize("metric", [hit_rate_at_k, recall_at_k, ndcg_at_k])
def test_metrics_are_zero_when_no_chunk_is_relevant(metric):
    """A question with no ground truth scores zero rather than dividing by zero."""
    assert metric(RANKED, set(), 5) == 0.0
