from app.reranker import RerankService

CHUNKS = [
    {"id": 1, "text": "peu pertinent"},
    {"id": 2, "text": "très pertinent"},
    {"id": 3, "text": "moyen"},
]


def test_assemble_sorts_by_score_and_truncates():
    out = RerankService._assemble(CHUNKS, [-2.0, 5.0, 0.0], top_n=2)
    assert [c["id"] for c in out] == [2, 3]
    assert out[0]["rerank_score"] > out[1]["rerank_score"]
    assert 0.0 <= out[0]["rerank_score"] <= 1.0


def test_assemble_does_not_mutate_inputs():
    RerankService._assemble(CHUNKS, [1.0, 2.0, 3.0], top_n=3)
    assert all("rerank_score" not in c for c in CHUNKS)


def test_assemble_empty():
    assert RerankService._assemble([], [], top_n=5) == []


def test_assemble_threshold_drops_irrelevant():
    out = RerankService._assemble(CHUNKS, [5.0, -6.0, -7.0], top_n=5, min_score=0.05)
    assert [c["id"] for c in out] == [1]


def test_assemble_threshold_always_keeps_the_best():
    """Even when everything scores low, return something rather than nothing."""
    out = RerankService._assemble(CHUNKS, [-6.0, -7.0, -8.0], top_n=5, min_score=0.05)
    assert len(out) == 1 and out[0]["id"] == 1
