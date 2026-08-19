"""Liveness and readiness must answer different questions.

Conflating them is how a deployment stays green while the service cannot serve:
the platform keeps routing traffic to a container whose database is gone.
"""

from __future__ import annotations

from tests.integration.conftest import Harness


async def test_health_reports_liveness_without_touching_the_database(harness: Harness):
    await harness.db.disconnect()

    response = await harness.client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_reports_ready_when_the_database_answers(harness: Harness):
    response = await harness.client.get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}


async def test_ready_returns_503_when_the_database_is_gone(harness: Harness):
    """The whole point: withhold traffic instead of failing in the user's face."""
    await harness.db.disconnect()

    response = await harness.client.get("/api/ready")

    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


async def test_every_response_carries_a_request_id(harness: Harness):
    response = await harness.client.get("/api/health")

    assert response.headers["x-request-id"]


async def test_an_incoming_request_id_is_preserved(harness: Harness):
    """A trace has to survive across service boundaries to be worth anything."""
    response = await harness.client.get(
        "/api/health", headers={"X-Request-ID": "trace-de-bout-en-bout"}
    )

    assert response.headers["x-request-id"] == "trace-de-bout-en-bout"


async def test_a_request_id_is_issued_even_for_a_rejected_request(harness: Harness):
    """404s and 413s are exactly the responses someone will ask you about."""
    response = await harness.client.get("/api/inexistant")

    assert response.status_code == 404
    assert response.headers["x-request-id"]
