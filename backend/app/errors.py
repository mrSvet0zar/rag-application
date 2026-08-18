"""Domain errors.

Services raise these; the API layer is responsible for mapping them to status
codes. Keeping `HTTPException` out of the services means they can be reused
(CLI, workers, tests) without dragging FastAPI along.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-facing failures."""


class UnreadableDocumentError(AppError):
    """The file could not be parsed, or contained no extractable text."""


class IngestionFailedError(AppError):
    """Chunking/embedding/storage failed after the document row was created."""


class NotFoundError(AppError):
    """A referenced resource does not exist."""


class UrlNotAllowedError(AppError):
    """The requested URL failed the SSRF / scheme checks."""


class UrlFetchError(AppError):
    """The remote page could not be fetched."""
