"""Document chunking.

Split out of the generation pipeline: ingestion needs to chunk text but has no
business depending on an LLM client, and chunking is worth testing on its own.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Splits raw document text into overlapping, embedding-sized chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        """Return non-empty, stripped chunks (empty input -> empty list)."""
        return [c.strip() for c in self._splitter.split_text(text) if c.strip()]
