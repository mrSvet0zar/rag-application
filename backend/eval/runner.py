"""Measure retrieval quality against the golden set.

Runs the *real* pipeline — real embedding model, real pgvector, real reranker —
because the point is to measure what production does, not a simplified stand-in.
Only the LLM is left out: this stage evaluates retrieval, which is what every
later answer depends on.

Ground truth is derived, not stored. The golden set names verbatim snippets; the
runner scans the indexed chunks for them and treats every chunk containing one as
relevant. That keeps the judgements valid across chunking strategies, which is
what makes the A/B chunking experiment possible.

Usage (from `backend/`):

    python -m eval.runner                          # current settings
    python -m eval.runner --no-rerank              # vector search alone
    python -m eval.runner --chunk-size 512 --chunk-overlap 100
    python -m eval.runner --compare                # dense/hybrid x rerank on/off
    python -m eval.runner --sweep                  # compare chunking strategies

Each configuration is indexed into its own database (`rag_eval_c1024_o200`…),
so switching configurations never mixes two indexes and re-running one is
instant.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from app.chunking import TextChunker
from app.config import Settings, get_settings
from app.embeddings import EmbeddingService
from app.ingestor import DocumentIngestor
from app.protocols import Embedder, Reranker
from app.reranker import RerankService
from app.retrieval import Retriever
from app.vector_db import Database
from eval.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_SET = EVAL_DIR / "golden_set.json"


@dataclass(frozen=True)
class Variant:
    """One configuration to measure."""

    name: str
    rerank: bool
    hybrid: bool
    chunk_size: int
    chunk_overlap: int
    k: int

    @property
    def index_name(self) -> str:
        """Database holding the index for this chunking configuration.

        Reranking is not part of the name: it happens at query time and does
        not change what is stored.
        """
        return f"rag_eval_c{self.chunk_size}_o{self.chunk_overlap}"


@dataclass
class Outcome:
    """Per-question result, kept so failures can be inspected individually."""

    question_id: str
    kind: str
    relevant: int
    retrieved: list[int]
    documents: list[str]
    latency_ms: float
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- indexing ----


def _with_database(url: str, name: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path=f"/{name}"))


async def _create_database_if_missing(url: str, name: str) -> None:
    admin = await asyncpg.connect(_with_database(url, "postgres"))
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", name
        )
        if not exists:
            # Identifier cannot be parameterised; it is built from our own config.
            await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()


async def _index_corpus(db: Database, embedder: Embedder, variant: Variant) -> int:
    """Index every corpus file, unless this database already holds them."""
    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(
            f"Corpus vide dans {CORPUS_DIR}. Lancez d'abord "
            "`python -m eval.fetch_corpus`."
        )

    existing = {doc["filename"] for doc in await db.list_documents()}
    expected = {path.stem for path in files}
    if expected <= existing:
        async with db.pool.acquire() as conn:
            return int(await conn.fetchval("SELECT count(*) FROM chunks"))

    chunker = TextChunker(variant.chunk_size, variant.chunk_overlap)
    ingestor = DocumentIngestor(db, embedder, chunker)

    print(f"Indexation de {len(files)} documents ({variant.index_name})…")
    total = 0
    for path in files:
        if path.stem in existing:
            continue
        text = path.read_text(encoding="utf-8")
        document = await ingestor.ingest(
            filename=path.stem,
            content_type="text/plain",
            size_bytes=len(text.encode()),
            text=text,
        )
        total += document["total_chunks"]
        print(f"  {path.stem}: {document['total_chunks']} chunks")

    async with db.pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM chunks"))


# ------------------------------------------------------------ ground truth ----


async def _load_chunks(db: Database) -> list[tuple[int, str]]:
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, text FROM chunks")
    return [(row["id"], row["text"]) for row in rows]


def ground_truth(chunks: list[tuple[int, str]], snippets: list[str]) -> set[int]:
    """Ids of every chunk containing one of the expected snippets."""
    return {
        chunk_id
        for chunk_id, text in chunks
        if any(snippet in text for snippet in snippets)
    }


# ------------------------------------------------------------- measurement ----


async def evaluate(variant: Variant, settings: Settings) -> list[Outcome]:
    """Index if needed, then run every golden question through the retriever."""
    await _create_database_if_missing(settings.database_url, variant.index_name)
    scoped = settings.model_copy(
        update={
            "database_url": _with_database(settings.database_url, variant.index_name),
            "chunk_size": variant.chunk_size,
            "chunk_overlap": variant.chunk_overlap,
            "hybrid_enabled": variant.hybrid,
        }
    )

    embedder: Embedder = EmbeddingService(scoped)
    reranker: Reranker | None = RerankService(scoped) if variant.rerank else None

    db = Database(scoped)
    await db.connect()
    try:
        await _index_corpus(db, embedder, variant)
        chunks = await _load_chunks(db)
        retriever = Retriever(db, embedder, reranker, scoped)

        golden = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
        outcomes: list[Outcome] = []

        # One untimed query first: the embedding and cross-encoder models load
        # lazily, and charging that to the first question would put several
        # seconds into the latency percentiles that no real request ever pays.
        await retriever.retrieve("question de préchauffage", variant.k)

        for question in golden["questions"]:
            relevant = ground_truth(chunks, question["snippets"])
            if not relevant:
                # A snippet that matches nothing would silently score 0 and drag
                # the average down for the wrong reason.
                raise SystemExit(
                    f"{question['id']}: aucun chunk ne contient les extraits attendus. "
                    "Le golden set et le corpus ont divergé."
                )

            started = time.perf_counter()
            results = await retriever.retrieve(question["question"], variant.k)
            latency_ms = (time.perf_counter() - started) * 1000
            retrieved = [row["id"] for row in results]
            documents = [row.get("filename") or "" for row in results]

            outcomes.append(
                Outcome(
                    question_id=question["id"],
                    kind=question["kind"],
                    relevant=len(relevant),
                    retrieved=retrieved,
                    documents=documents,
                    latency_ms=latency_ms,
                    metrics={
                        "hit@k": hit_rate_at_k(retrieved, relevant, variant.k),
                        # Separates "found the right passage" from "found the
                        # right article but the wrong passage in it".
                        "doc_hit@k": float(question["document"] in documents),
                        "recall@k": recall_at_k(retrieved, relevant, variant.k),
                        "precision@k": precision_at_k(retrieved, relevant, variant.k),
                        "mrr": reciprocal_rank(retrieved, relevant),
                        "ndcg@k": ndcg_at_k(retrieved, relevant, variant.k),
                    },
                )
            )
        return outcomes
    finally:
        await db.disconnect()


METRIC_NAMES = ("hit@k", "doc_hit@k", "recall@k", "precision@k", "mrr", "ndcg@k")


def aggregate(outcomes: list[Outcome]) -> dict[str, float]:
    summary = {
        name: statistics.fmean(o.metrics[name] for o in outcomes) for name in METRIC_NAMES
    }
    latencies = sorted(o.latency_ms for o in outcomes)
    summary["p50_ms"] = statistics.median(latencies)
    # 95th percentile, nearest-rank: honest on a sample this small.
    summary["p95_ms"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    return summary


def _row(label: str, summary: dict[str, float]) -> str:
    cells = " ".join(f"{summary[name]:>11.3f}" for name in METRIC_NAMES)
    return f"{label:<22}{cells} {summary['p50_ms']:>9.0f} {summary['p95_ms']:>9.0f}"


def sweep_header(k_label: str) -> None:
    header = " ".join(f"{name:>11}" for name in METRIC_NAMES)
    print(f"{'':<22}{header} {'p50 (ms)':>9} {'p95 (ms)':>9}  {k_label}")
    print("-" * (22 + 12 * len(METRIC_NAMES) + 32))


def report(title: str, outcomes: list[Outcome], k: int) -> None:
    header = " ".join(f"{name.replace('@k', f'@{k}'):>11}" for name in METRIC_NAMES)
    print(f"\n{title}")
    print(f"{'':<22}{header} {'p50 (ms)':>9} {'p95 (ms)':>9}")
    print("-" * (22 + 12 * len(METRIC_NAMES) + 20))
    print(_row("global", aggregate(outcomes)))
    for kind in sorted({o.kind for o in outcomes}):
        subset = [o for o in outcomes if o.kind == kind]
        print(_row(f"  {kind} ({len(subset)})", aggregate(subset)))

    missed = [o for o in outcomes if o.metrics["hit@k"] == 0.0]
    if missed:
        print(
            f"\nAucun passage pertinent dans le top {k} : "
            + ", ".join(f"{o.question_id} ({o.kind})" for o in missed)
        )


# -------------------------------------------------------------------- main ----


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    parser.add_argument("--no-hybrid", action="store_true", help="dense retrieval only")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="compare chunking strategies at fixed k and fixed context budget",
    )
    parser.add_argument(
        "--budget-chars",
        type=int,
        default=5120,
        help="context budget for the sweep (default: 5 x 1024)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run dense/hybrid x with/without reranking, side by side",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.sweep:
        return await _sweep(settings, args.k, args.budget_chars)

    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap

    if args.compare:
        # The four combinations that matter, so each stage's contribution is
        # visible on its own rather than only in aggregate.
        variants = [
            Variant("vectoriel seul", False, False, chunk_size, chunk_overlap, args.k),
            Variant("vectoriel + rerank", True, False, chunk_size, chunk_overlap, args.k),
            Variant("hybride seul", False, True, chunk_size, chunk_overlap, args.k),
            Variant("hybride + rerank", True, True, chunk_size, chunk_overlap, args.k),
        ]
    else:
        rerank = settings.rerank_enabled and not args.no_rerank
        hybrid = settings.hybrid_enabled and not args.no_hybrid
        label = ("hybride" if hybrid else "vectoriel") + (
            " + rerank" if rerank else " seul"
        )
        variants = [Variant(label, rerank, hybrid, chunk_size, chunk_overlap, args.k)]

    print(f"corpus : {CORPUS_DIR}  |  chunk_size={chunk_size} overlap={chunk_overlap}")
    for variant in variants:
        outcomes = await evaluate(variant, settings)
        report(
            f"{variant.name}  (k={variant.k}, {len(outcomes)} questions)",
            outcomes,
            variant.k,
        )
    return 0


# Overlap is kept at ~20% of the chunk size so the comparison varies one thing.
SWEEP_CONFIGS: tuple[tuple[int, int], ...] = (
    (256, 50),
    (512, 100),
    (1024, 200),
    (2048, 400),
)


async def _sweep(settings: Settings, k: int, budget_chars: int) -> int:
    """Compare chunking strategies, holding retrieval fixed at its best setting.

    Reported twice, because a single view would mislead:

    * **At constant k** — the usual comparison, but unfair on context: five
      256-character chunks give the LLM a quarter of the text that five
      1024-character chunks do.
    * **At constant context budget** — k is scaled so every configuration hands
      the generator roughly the same number of characters, which is the
      comparison that matches what the LLM actually receives.

    Only `hit@k`, `doc_hit@k` and `mrr` are strictly comparable across chunk
    sizes: `recall@k` and `precision@k` are relative to the number of relevant
    chunks, and that count itself changes with the chunking.
    """
    print(f"corpus : {CORPUS_DIR}  |  hybride + reranking")
    print()

    print(f"=== à k constant (k={k}) ===")
    sweep_header("")
    for chunk_size, overlap in SWEEP_CONFIGS:
        variant = Variant(f"{chunk_size}/{overlap}", True, True, chunk_size, overlap, k)
        outcomes = await evaluate(variant, settings)
        print(_row(f"  {variant.name}", aggregate(outcomes)))

    print()
    print(f"=== à budget de contexte constant (~{budget_chars} caractères) ===")
    sweep_header("k")
    for chunk_size, overlap in SWEEP_CONFIGS:
        scaled = max(1, round(budget_chars / chunk_size))
        variant = Variant(
            f"{chunk_size}/{overlap}", True, True, chunk_size, overlap, scaled
        )
        outcomes = await evaluate(variant, settings)
        print(_row(f"  {variant.name}", aggregate(outcomes)) + f" {scaled:>9}")
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
