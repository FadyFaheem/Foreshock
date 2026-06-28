"""Tests for the /health endpoint."""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "foreshock-api"


def test_health_post_not_allowed(client):
    resp = client.post("/health")
    assert resp.status_code == 405
