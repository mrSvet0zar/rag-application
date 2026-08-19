"""Application configuration, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings sourced from the backend/.env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # --- LLM (Anthropic) ---
    anthropic_api_key: str = ""
    chat_model: str = "claude-sonnet-5"
    max_tokens: int = 1024

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    pgvector_dimension: int = 384

    # --- App ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    # Ceiling on any request body (uploads and fetched pages alike). Chosen
    # to stay well under the container's memory budget, which already holds
    # two ML models.
    max_upload_bytes: int = 10 * 1024 * 1024
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- RAG ---
    # 512/100 chosen by measurement, not by habit: see docs/EVALUATION.md.
    # It beats 1024/200 on hit@k and doc_hit@k while halving retrieval
    # latency, because the cross-encoder scores passages half as long.
    chunk_size: int = 512
    chunk_overlap: int = 100
    min_relevance_score: float = 0.25
    top_k_retrieval: int = 5

    # --- Hybrid retrieval (lexical + vector) ---
    # Vector search is blind to exact terms (acronyms, proper nouns, foreign
    # titles); a lexical pass covers exactly that. The two rankings are merged
    # with Reciprocal Rank Fusion, which combines *ranks* and so needs no
    # normalisation between two incomparable score scales.
    hybrid_enabled: bool = True
    lexical_candidates: int = 20
    # RRF damping constant. 60 is the value from the original paper and is the
    # conventional default; higher flattens the contribution of top ranks.
    rrf_k: int = 60

    # --- Reranking (cross-encoder) ---
    # Retrieve `rerank_candidates` chunks by vector similarity, then re-score
    # them with a cross-encoder and keep the best `k`. Set rerank_enabled=False
    # to skip it (e.g. on a memory-constrained host: it loads a 2nd model).
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidates: int = 20
    # How many candidates the cross-encoder is allowed to score. Distinct from
    # `rerank_candidates`, which is per retriever: fusing two rankings of 20
    # yields up to 40, and truncating that back to 20 discards most of what the
    # lexical pass contributed before the reranker ever sees it.
    rerank_max_candidates: int = 40
    # Drop chunks the cross-encoder scores below this (keeps at least the top 1),
    # so obviously irrelevant passages aren't fed to the LLM or shown as sources.
    rerank_min_score: float = 0.05

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list (env stores a comma-separated string)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def demo_mode(self) -> bool:
        """True when no Anthropic key is set -> answers use a local template."""
        return not self.anthropic_api_key.strip()


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
