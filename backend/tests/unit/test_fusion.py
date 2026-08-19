"""RRF decides the final ordering of hybrid retrieval, so its behaviour is
pinned rather than assumed."""

from __future__ import annotations

from app.fusion import fuse_rows, reciprocal_rank_fusion


def test_an_item_found_by_both_retrievers_outranks_one_found_by_only_one():
    """This is the whole point of fusing: agreement beats a narrow single win."""
    vector = [1, 2, 3]
    lexical = [4, 2, 5]

    assert reciprocal_rank_fusion([vector, lexical])[0] == 2


def test_a_single_ranking_is_returned_unchanged():
    assert reciprocal_rank_fusion([[7, 3, 9]]) == [7, 3, 9]


def test_empty_input_yields_empty_output():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_items_from_either_list_all_survive():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]])
    assert set(fused) == {1, 2, 3, 4}


def test_ties_break_deterministically_on_the_id():
    """Run-to-run reshuffling would surface as noise in the evaluation."""
    first = reciprocal_rank_fusion([[10, 20], [20, 10]])
    second = reciprocal_rank_fusion([[10, 20], [20, 10]])

    assert first == second == [10, 20]


def test_k_controls_how_much_a_top_rank_is_worth():
    """Small k makes rank 1 dominant; large k lets deeper agreement win.

    id 1 is first in one list only (1/(k+1)); id 5 is fourth in both
    (2/(k+4)). The crossover sits at k+2 = 4, so k=1 favours the lone top hit
    and the default k=60 favours the agreement.
    """
    rankings = [[1, 2, 3, 5], [6, 7, 8, 5]]

    assert reciprocal_rank_fusion(rankings, k=1)[0] == 1
    assert reciprocal_rank_fusion(rankings, k=60)[0] == 5


def test_a_repeated_id_inside_one_ranking_is_counted_once():
    """Otherwise a duplicate would score as if two retrievers had agreed."""
    with_duplicate = reciprocal_rank_fusion([[1, 1, 1], [2, 3, 4]])

    assert with_duplicate[0] == 1  # still first, on its best rank alone
    assert with_duplicate == reciprocal_rank_fusion([[1], [2, 3, 4]])


def test_fuse_rows_merges_fields_from_both_retrievers():
    vector = [{"id": 1, "text": "a", "similarity": 0.8}]
    lexical = [{"id": 1, "text": "a", "lexical_score": 0.3}]

    fused = fuse_rows([vector, lexical])

    assert len(fused) == 1
    assert fused[0]["similarity"] == 0.8
    assert fused[0]["lexical_score"] == 0.3


def test_fuse_rows_keeps_rows_seen_by_a_single_retriever():
    vector = [{"id": 1, "similarity": 0.8}]
    lexical = [{"id": 2, "lexical_score": 0.3}]

    fused = fuse_rows([vector, lexical])

    assert {row["id"] for row in fused} == {1, 2}
    assert "lexical_score" not in fused[0] or "similarity" not in fused[1]
