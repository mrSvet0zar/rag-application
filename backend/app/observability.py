"""Structured logging and request correlation.

Line-oriented logs are fine to read over someone's shoulder and useless in
production: nothing can query "every failed upload in the last hour" from prose.
These emit JSON, so a log backend can index the fields.

The other half is correlation. A single question fans out into embedding,
vector search, lexical search, reranking and an LLM call, each logging
separately; without a shared id there is no way to reassemble one user's request
from an interleaved stream. A contextvar carries that id — contextvars follow
`await` boundaries and stay per-task, which a module-level global would not.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# Set per request by the middleware; empty outside a request (startup, scripts).
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Attributes LogRecord always carries. Anything else was passed via `extra`
# and is application context worth emitting.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id

        # Whatever the call site passed as extra=... — the useful part.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(fmt: str = "json", level: int = logging.INFO) -> None:
    """Install the formatter on the root logger.

    Uvicorn's own loggers propagate to root, so its access and error lines end
    up in the same structured stream rather than in a second, differently
    shaped one.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


@contextmanager
def timed(store: dict[str, float], key: str) -> Iterator[None]:
    """Record how long a block took, in milliseconds, under `key`.

    Used to break a request down by stage. Knowing a question took two seconds
    is not actionable; knowing the cross-encoder spent 1.8 of them is.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - started) * 1000, 2)
