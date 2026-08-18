"""Local embedding service using sentence-transformers (no API key required).

The model is loaded lazily on first use so that importing this module (e.g. in
tests) stays cheap. Encoding is CPU-bound, so callers should run it in a thread
pool when inside an async request (see main.py).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a sentence-transformers model to produce normalized embeddings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        """Load the model on first use (thread-safe, idempotent)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Imported here so the heavy torch import only happens when
                    # embeddings are actually needed.
                    from sentence_transformers import SentenceTransformer

                    logger.info(
                        "Loading embedding model '%s' ...",
                        self._settings.embedding_model,
                    )
                    self._model = SentenceTransformer(self._settings.embedding_model)
                    logger.info("Embedding model loaded.")
        return self._model

    @property
    def dimension(self) -> int:
        return self._settings.pgvector_dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts."""
        if not texts:
            return []
        model = self._ensure_model()
        # normalize_embeddings=True -> cosine similarity == dot product.
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        model = self._ensure_model()
        vector = model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()
