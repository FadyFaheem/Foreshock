"""Tests for the AI blueprint (status, observability, evals, input validation).

These are fast and degrade-friendly: endpoints that need the database skip
themselves when it is unavailable. The LLM-heavy diagnose/agent paths are not
exercised here (they are covered by scripts/run_evals.py).
"""

import pytest


def test_ai_status(client):
    resp = client.get("/api/ai/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert {"db", "llm", "kb_size", "model", "embed_model"} <= data.keys()
    assert isinstance(data["db"], bool)
    assert isinstance(data["llm"], bool)


def test_diagnose_requires_input(client):
    resp = client.post("/api/diagnose", data={})
    # 400 when the engine is loaded but no input; 503 if the model is missing.
    assert resp.status_code in (400, 503)


def test_observability_shape(client):
    resp = client.get("/api/observability")
    if resp.status_code == 503:
        pytest.skip("database unavailable")
    assert resp.status_code == 200
    data = resp.get_json()
    assert {"summary", "by_operation", "recent"} <= data.keys()
    assert {"calls", "avg_latency_ms", "total_tokens"} <= data["summary"].keys()


def test_evals_latest(client):
    resp = client.get("/api/evals")
    if resp.status_code == 503:
        pytest.skip("database unavailable")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), dict)


def test_work_orders_list(client):
    resp = client.get("/api/work_orders")
    if resp.status_code == 503:
        pytest.skip("database unavailable")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)
