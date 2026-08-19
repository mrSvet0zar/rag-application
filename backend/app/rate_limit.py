"""Token-bucket rate limiting for the endpoints that cost money.

Not every route needs protecting: listing documents is cheap. Asking a question
calls a paid LLM, and uploading runs an embedding model over a whole file — so
those are what an unprotected public deployment leaks money through.

A token bucket rather than a fixed window: a fixed window lets a caller spend
its whole quota in the last second of one window and again in the first second
of the next, which is twice the intended rate at the worst possible moment. A
bucket refills continuously, so a burst is allowed only up to its capacity and
the long-run rate is exactly the refill rate.

Buckets live in this process. That is honest for a single instance and stated
plainly rather than pretended otherwise: with several replicas each enforces the
limit separately, and the effective limit is multiplied by the replica count.
Fixing that properly means shared state (Redis), which is not worth its
operational weight here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """A bucket that refills at `rate` tokens per second, capped at `capacity`."""

    capacity: float
    rate: float
    tokens: float = field(init=False)
    updated_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.updated_at = time.monotonic()

    def take(self, now: float | None = None) -> bool:
        """Consume one token. False when the bucket is empty."""
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated_at = now

        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True

    def retry_after(self, now: float | None = None) -> int:
        """Whole seconds until one token is available (at least 1)."""
        now = time.monotonic() if now is None else now
        missing = max(0.0, 1.0 - self.tokens)
        return max(1, int(missing / self.rate + 0.999)) if self.rate else 1


class RateLimiter:
    """Per-client buckets, created on demand and pruned when idle."""

    def __init__(self, capacity: int, per_seconds: float) -> None:
        self._capacity = float(capacity)
        self._rate = capacity / per_seconds if per_seconds else float(capacity)
        self._buckets: dict[str, TokenBucket] = {}
        self._last_prune = time.monotonic()
        # A full bucket is indistinguishable from no bucket, so one that has had
        # time to refill completely can be dropped without changing behaviour.
        self._idle_ttl = per_seconds * 2

    def allow(self, key: str) -> tuple[bool, int]:
        """(allowed, retry_after_seconds) for one request from `key`."""
        self._maybe_prune()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(capacity=self._capacity, rate=self._rate)
            self._buckets[key] = bucket

        if bucket.take():
            return True, 0
        return False, bucket.retry_after()

    def _maybe_prune(self) -> None:
        """Drop idle buckets so a spray of one-off clients cannot grow memory."""
        now = time.monotonic()
        if now - self._last_prune < self._idle_ttl:
            return
        self._last_prune = now
        cutoff = now - self._idle_ttl
        self._buckets = {
            key: bucket
            for key, bucket in self._buckets.items()
            if bucket.updated_at > cutoff
        }
