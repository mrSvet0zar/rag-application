"""Rate limiting is what stands between a public endpoint and a drained API
budget, so its arithmetic is pinned rather than trusted.

Time is injected instead of slept through: a test that waits a real minute to
prove a refill is a test nobody runs.
"""

from __future__ import annotations

from app.rate_limit import RateLimiter, TokenBucket


def test_a_bucket_allows_up_to_its_capacity_then_refuses():
    bucket = TokenBucket(capacity=3, rate=1.0)
    now = bucket.updated_at

    assert [bucket.take(now) for _ in range(4)] == [True, True, True, False]


def test_tokens_come_back_over_time():
    bucket = TokenBucket(capacity=2, rate=1.0)  # one token per second
    now = bucket.updated_at
    bucket.take(now)
    bucket.take(now)

    assert bucket.take(now) is False
    assert bucket.take(now + 1.0) is True


def test_refill_is_capped_at_capacity():
    """An idle client must not accumulate an unlimited burst."""
    bucket = TokenBucket(capacity=2, rate=1.0)
    now = bucket.updated_at

    assert [bucket.take(now + 3600) for _ in range(3)] == [True, True, False]


def test_a_burst_costs_exactly_its_capacity_not_more():
    """The bucket is what a fixed window gets wrong: a fixed window would allow
    a full quota at the end of one window and again at the start of the next."""
    bucket = TokenBucket(capacity=10, rate=10 / 60)  # 10 per minute
    now = bucket.updated_at

    allowed = sum(1 for _ in range(20) if bucket.take(now))
    # Half a minute later, half the quota has come back — not all of it.
    allowed_after = sum(1 for _ in range(20) if bucket.take(now + 30))

    assert allowed == 10
    assert allowed_after == 5


def test_retry_after_is_at_least_one_second():
    bucket = TokenBucket(capacity=1, rate=1.0)
    bucket.take()

    assert bucket.retry_after() >= 1


def test_clients_are_limited_independently():
    """One noisy caller must not lock everyone else out."""
    limiter = RateLimiter(capacity=1, per_seconds=60)

    assert limiter.allow("1.2.3.4")[0] is True
    assert limiter.allow("1.2.3.4")[0] is False
    assert limiter.allow("5.6.7.8")[0] is True


def test_a_rejected_request_reports_when_to_retry():
    limiter = RateLimiter(capacity=1, per_seconds=60)
    limiter.allow("client")

    allowed, retry_after = limiter.allow("client")

    assert allowed is False
    assert 1 <= retry_after <= 60


def test_idle_buckets_are_pruned():
    """A spray of one-off clients must not grow memory without bound.

    The buckets are aged by hand rather than waited out: a bucket that has had
    time to refill fully is indistinguishable from no bucket at all, which is
    exactly what makes dropping it safe.
    """
    limiter = RateLimiter(capacity=5, per_seconds=60)
    for index in range(50):
        limiter.allow(f"client-{index}")

    for bucket in limiter._buckets.values():
        bucket.updated_at -= 3600
    limiter._last_prune -= 3600  # make the next call due for a prune

    limiter.allow("client-actif")

    assert set(limiter._buckets) == {"client-actif"}


def test_a_recently_seen_bucket_survives_a_prune():
    """Pruning an active client would hand it a fresh quota."""
    limiter = RateLimiter(capacity=1, per_seconds=60)
    limiter.allow("actif")

    limiter._last_prune -= 3600  # force a prune on the next call
    limiter.allow("autre")

    assert limiter.allow("actif")[0] is False, "son quota doit être conservé"
