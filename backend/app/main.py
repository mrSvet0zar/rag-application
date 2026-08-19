"""Application factory.

`create_app` builds a fully wired FastAPI app. Tests call it with their own
settings and a services builder that swaps in fast, deterministic doubles; the
module-level `app` is what uvicorn serves in production.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, conversations, documents, health, stats
from app.config import Settings, get_settings
from app.errors import (
    AppError,
    IngestionFailedError,
    NotFoundError,
    PayloadTooLargeError,
    UnreadableDocumentError,
    UrlFetchError,
    UrlNotAllowedError,
)
from app.middleware import (
    MaxBodySizeMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from app.observability import configure_logging
from app.rate_limit import RateLimiter
from app.services import Services, build_services

logger = logging.getLogger("rag")

# Domain error -> HTTP status. Ordered: the first matching class wins, so put
# subclasses before their parents if that ever applies.
# Endpoints whose cost is an LLM call or a model inference, not a query.
COSTLY_PREFIXES = ("/api/chat", "/api/documents/upload", "/api/documents/import-url")

_ERROR_STATUS: tuple[tuple[type[AppError], int], ...] = (
    (NotFoundError, 404),
    (PayloadTooLargeError, 413),
    (UnreadableDocumentError, 400),
    (UrlNotAllowedError, 400),
    (UrlFetchError, 400),
    (IngestionFailedError, 500),
)


def _status_for(exc: AppError) -> int:
    for error_type, status in _ERROR_STATUS:
        if isinstance(exc, error_type):
            return status
    return 500


def create_app(
    settings: Settings | None = None,
    services_builder: Callable[[Settings], Services] = build_services,
) -> FastAPI:
    """Build the application. Pure wiring — no I/O until startup."""
    settings = settings or get_settings()
    configure_logging(settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = services_builder(settings)
        await services.db.connect()
        app.state.services = services

        # A background ingestion interrupted by a restart cannot resume, so its
        # row would sit in `processing` forever. Better an honest failure.
        abandoned = await services.db.fail_stale_processing()
        if abandoned:
            logger.warning(
                "marked abandoned ingestions as failed",
                extra={"event": "ingestion.reconciled", "documents": abandoned},
            )
        logger.info(
            "Startup complete (demo_mode=%s, rerank=%s).",
            settings.demo_mode,
            services.reranker is not None,
        )
        try:
            yield
        finally:
            await services.db.disconnect()

    app = FastAPI(title="RAG Application API", version="1.0.0", lifespan=lifespan)

    # Added before CORS so it ends up *inside* it: oversized requests are
    # rejected before any parsing, but the 413 still carries CORS headers.
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_upload_bytes)

    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            limiter=RateLimiter(
                settings.rate_limit_requests, settings.rate_limit_window_seconds
            ),
            prefixes=COSTLY_PREFIXES,
            trust_proxy_headers=settings.trust_proxy_headers,
        )

    # Added last, so it wraps everything: even a request rejected by the body
    # guard or the rate limiter gets an id and an access log line.
    app.add_middleware(RequestContextMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Single place translating domain failures into responses.

        Keeps the `{"detail": ...}` shape FastAPI uses for HTTPException so
        clients see one consistent error envelope.
        """
        status = _status_for(exc)
        if status >= 500:
            logger.exception("Unhandled domain error", exc_info=exc)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    for router in (
        health.router,
        documents.router,
        chat.router,
        conversations.router,
        stats.router,
    ):
        app.include_router(router, prefix="/api")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(app, host=_settings.api_host, port=_settings.api_port)
