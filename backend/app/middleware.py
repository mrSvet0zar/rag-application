"""ASGI middleware.

`MaxBodySizeMiddleware` has to be ASGI-level rather than a route dependency:
FastAPI fully parses a multipart body *before* the endpoint function runs, so a
size check inside the endpoint would fire only after the whole upload had
already been received and spooled to disk.
"""

from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability import request_id_var

_KB = 1024
_MB = 1024 * _KB

# Probed every few seconds by the platform; logged at DEBUG so they do not
# bury the requests a human cares about.
_QUIET_PATHS = frozenset({"/api/health", "/api/ready"})


def _human(size: int) -> str:
    return f"{size // _MB} Mo" if size >= _MB else f"{size // _KB} Ko"


class MaxBodySizeMiddleware:
    """Reject request bodies larger than `max_bytes`.

    Two layers, because either signal can be absent or dishonest:

    * `Content-Length`, when declared, is checked before a single byte of body
      is read — the cheap path that stops most oversized uploads outright.
    * Otherwise (chunked transfer, or a client that lies) the body is counted as
      it streams and the request is cut short the moment it goes over.

    The oversize condition is reported by *substituting the response* rather
    than by raising. Raising does not work in either layer: this middleware sits
    outside Starlette's ExceptionMiddleware, so an exception here would surface
    as a 500; and an exception raised while the body streams is swallowed by
    Starlette's multipart parser, which reports it as a generic 400 "error
    parsing the body". Overriding the response keeps the status honest whatever
    the application does with the truncated stream.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = self._declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        too_large = False
        forwarded_start = False
        substituted = False

        async def counting_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    # End the stream instead of feeding more: the application
                    # gets a truncated body, fails or succeeds as it likes, and
                    # its response is replaced below.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def limited_send(message: Message) -> None:
            nonlocal forwarded_start, substituted
            # Once the application has put a response on the wire it is too
            # late to substitute anything, so only override before its first
            # byte — and from then on swallow everything it emits, or the
            # client would receive two responses.
            if too_large and not forwarded_start:
                if not substituted:
                    await self._reject(send)
                    substituted = True
                return
            if message["type"] == "http.response.start":
                forwarded_start = True
            await send(message)

        await self.app(scope, counting_receive, limited_send)

    async def _reject(self, send: Send) -> None:
        payload = json.dumps({"detail": self._message()}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    def _declared_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    def _message(self) -> str:
        return f"Requête trop volumineuse (limite : {_human(self.max_bytes)})."


class RequestContextMiddleware:
    """Tag every request with an id, and log how it went.

    The id is taken from `X-Request-ID` when a proxy or client already set one,
    so a trace survives across service boundaries, and echoed back on the
    response — which is what makes a user's bug report ("request abc123 failed")
    actionable.

    ASGI-level rather than a FastAPI middleware so the access log covers
    responses that never reach a route: 404s, validation errors, and the 413s
    the body-size guard produces.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._incoming_id(scope) or uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500  # if the app raises, nothing sets this

        async def tagged_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, tagged_send)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            # Health checks fire constantly and would drown everything else.
            level = logging.DEBUG if scope.get("path") in _QUIET_PATHS else logging.INFO
            logging.getLogger("rag.access").log(
                level,
                "%s %s -> %s",
                scope.get("method", "?"),
                scope.get("path", "?"),
                status,
                extra={
                    "http_method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)

    @staticmethod
    def _incoming_id(scope: Scope) -> str | None:
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                # Bounded: an unbounded header would end up in every log line.
                return value.decode("latin-1")[:64] or None
        return None
