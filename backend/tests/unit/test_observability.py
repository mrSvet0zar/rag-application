"""Structured logging is only useful if it is actually parseable and correlated."""

from __future__ import annotations

import json
import logging

from app.observability import JsonFormatter, request_id_var, timed


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="quelque chose %s",
        args=("est arrivé",),
        exc_info=None,
    )
    record.__dict__.update(extra)
    return record


def test_a_record_renders_as_one_json_object():
    payload = json.loads(JsonFormatter().format(_record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "rag.test"
    assert payload["message"] == "quelque chose est arrivé"
    assert "ts" in payload


def test_extra_fields_are_emitted_as_top_level_keys():
    """That is what makes a log queryable rather than merely readable."""
    payload = json.loads(JsonFormatter().format(_record(status=413, duration_ms=12.5)))

    assert payload["status"] == 413
    assert payload["duration_ms"] == 12.5


def test_the_request_id_is_attached_when_one_is_set():
    token = request_id_var.set("abc123")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(token)

    assert payload["request_id"] == "abc123"


def test_no_request_id_key_outside_a_request():
    """Startup and scripts log too; an empty id would be noise in every line."""
    assert "request_id" not in json.loads(JsonFormatter().format(_record()))


def test_exceptions_are_serialised_into_the_object():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_a_message_containing_quotes_still_produces_valid_json():
    record = _record()
    record.msg = 'il a dit "bonjour" \\ et est parti'
    record.args = ()

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == 'il a dit "bonjour" \\ et est parti'


def test_timed_records_a_duration_under_its_key():
    stages: dict[str, float] = {}

    with timed(stages, "retrieval_ms"):
        pass

    assert "retrieval_ms" in stages
    assert stages["retrieval_ms"] >= 0.0


def test_timed_records_the_duration_even_when_the_block_raises():
    """A failed stage is exactly the one whose timing matters."""
    stages: dict[str, float] = {}

    try:
        with timed(stages, "generation_ms"):
            raise RuntimeError("échec")
    except RuntimeError:
        pass

    assert "generation_ms" in stages
