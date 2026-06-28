"""Tests for v2 (health indicator) and v3 (live stream) endpoints.

Health endpoints skip if the health model hasn't been trained; the stream
status endpoint always responds (reporting whether Kafka is reachable).
"""

import pytest


def test_health_trend(client):
    resp = client.get("/api/health/trend")
    if resp.status_code == 503:
        pytest.skip("health model not trained; run scripts/train_health.py")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "threshold" in data and "points" in data
    assert len(data["points"]) > 0
    assert {"i", "error", "smooth", "phase"} <= data["points"][0].keys()


def test_health_embedding(client):
    resp = client.get("/api/health/embedding")
    if resp.status_code == 503:
        pytest.skip("health model not trained")
    assert resp.status_code == 200
    pts = resp.get_json()["points"]
    assert len(pts) > 0
    assert {"x", "y", "condition"} <= pts[0].keys()


def test_health_sample(client):
    resp = client.get("/api/health/sample/normal")
    if resp.status_code == 503:
        pytest.skip("health model not trained")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "mean_error" in data and "threshold" in data and "errors" in data


def test_stream_status(client):
    resp = client.get("/api/stream/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "kafka" in data and isinstance(data["kafka"], bool)
    assert data["topic"]
