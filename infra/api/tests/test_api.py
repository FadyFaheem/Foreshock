"""Tests for the Foreshock API blueprint (/api/*).

Skipped automatically if the model/samples have not been trained yet.
"""

import io

import pytest


@pytest.fixture(autouse=True)
def _require_engine(client):
    """Skip API tests when the engine could not be loaded (no trained model)."""
    if client.get("/api/samples").status_code == 503:
        pytest.skip("model not trained; run scripts/train.py")


def test_list_samples(client):
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list) and len(data) >= 1
    for item in data:
        assert {"id", "condition", "label"} <= item.keys()


def test_get_signal_shapes(client):
    sample_id = client.get("/api/samples").get_json()[0]["id"]
    resp = client.get(f"/api/signal/{sample_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["waveform"]["t"]) == len(data["waveform"]["x"])
    assert len(data["spectrum"]["f"]) == len(data["spectrum"]["mag"])
    assert len(data["envelope"]["f"]) == len(data["envelope"]["mag"])
    assert set(data["fault_frequencies"]) == {"BPFO", "BPFI", "BSF", "FTF"}


def test_get_signal_unknown_id_404(client):
    assert client.get("/api/signal/does-not-exist").status_code == 404


def test_predict_sample_id(client):
    sample = client.get("/api/samples").get_json()[0]
    resp = client.post("/api/predict", data={"sample_id": sample["id"]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] == sample["condition"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) >= 1
    assert len(data["features"]) == 16


def test_predict_requires_input(client):
    assert client.post("/api/predict", data={}).status_code == 400


def test_random_test(client):
    resp = client.post("/api/random_test", data={"condition": "random", "noise": "0"})
    assert resp.status_code == 200
    d = resp.get_json()
    assert {"actual", "prediction", "correct", "confidence", "waveform"} <= d.keys()
    assert isinstance(d["correct"], bool)
    assert len(d["waveform"]["t"]) == len(d["waveform"]["x"])


def test_random_test_specific_condition(client):
    resp = client.post("/api/random_test", data={"condition": "ball", "noise": "0.1"})
    assert resp.status_code == 200
    assert resp.get_json()["actual"] == "ball"


def test_inject_base_and_inject(client):
    base = client.get("/api/inject/base")
    assert base.status_code == 200
    sig = base.get_json()["signal"]
    assert len(sig) >= 2048
    resp = client.post(
        "/api/inject",
        json={"signal": sig, "points": [100, 500, 1500], "amplitude": 2.5},
    )
    assert resp.status_code == 200
    d = resp.get_json()
    assert {"caught", "prediction", "confidence", "waveform"} <= d.keys()
    assert isinstance(d["caught"], bool)


def test_predict_csv_upload(client):
    csv_bytes = ("\n".join("0.0" for _ in range(4096))).encode()
    data = {"file": (io.BytesIO(csv_bytes), "signal.csv")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "prediction" in resp.get_json()
