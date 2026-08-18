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
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- RAG ---
    chunk_size: int = 1024
    chunk_overlap: int = 200
    min_relevance_score: float = 0.25
    top_k_retrieval: int = 5

    # --- Reranking (cross-encoder) ---
    # Retrieve `rerank_candidates` chunks by vector similarity, then re-score
    # them with a cross-encoder and keep the best `k`. Set rerank_enabled=False
    # to skip it (e.g. on a memory-constrained host: it loads a 2nd model).
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidates: int = 20
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
