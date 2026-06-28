"""v2 health-indicator API: reconstruction-error trend + 2-D embedding.

Serves the precomputed run-to-failure timeline and embedding from
``models/health.npz`` and can score any bundled sample against the autoencoder.
"""

from __future__ import annotations

import numpy as np
from flask import Blueprint, jsonify

from engine import get_engine
from src import config, data_loader
from src import health as health_engine
from src.features import extract_features_batch

health_bp = Blueprint("health", __name__, url_prefix="/api/health")

_data: dict | None = None
_model: health_engine.HealthModel | None = None


def _load():
    global _data, _model
    if _data is None and config.MODELS_DIR.joinpath("health.npz").exists():
        npz = np.load(config.MODELS_DIR / "health.npz", allow_pickle=True)
        _data = {k: npz[k] for k in npz.files}
    if _model is None and config.MODELS_DIR.joinpath("health_ae.joblib").exists():
        _model = health_engine.load(config.MODELS_DIR / "health_ae.joblib")
    return _data, _model


def _label(condition: str) -> str:
    return config.CONDITION_LABELS.get(condition, condition)


@health_bp.get("/trend")
def trend():
    data, _ = _load()
    if data is None:
        return jsonify(error="Health model not trained. Run scripts/train_health.py."), 503
    err = data["timeline_error"]
    sm = data["timeline_smooth"]
    ph = data["timeline_phase"]
    points = [
        {"i": i, "error": float(err[i]), "smooth": float(sm[i]), "phase": str(ph[i])}
        for i in range(len(err))
    ]
    return jsonify(
        source=str(data["source"]),
        threshold=float(data["threshold"]),
        alarm_index=int(data["alarm_index"]),
        points=points,
    )


@health_bp.get("/embedding")
def embedding():
    data, _ = _load()
    if data is None:
        return jsonify(error="Health model not trained. Run scripts/train_health.py."), 503
    ex, ey, ec = data["embed_x"], data["embed_y"], data["embed_condition"]
    points = [
        {"x": float(ex[i]), "y": float(ey[i]), "condition": str(ec[i]),
         "label": _label(str(ec[i]))}
        for i in range(len(ex))
    ]
    return jsonify(points=points)


@health_bp.get("/sample/<sample_id>")
def sample(sample_id: str):
    """Score one bundled sample: per-window reconstruction error + embedding."""
    _, model = _load()
    eng = get_engine()
    if model is None:
        return jsonify(error="Health model not trained. Run scripts/train_health.py."), 503
    if eng is None or sample_id not in eng.index:
        return jsonify(error=f"Unknown sample id: {sample_id}"), 404

    i = eng.index[sample_id]
    windows = data_loader.segment(eng.signals[i])
    X = extract_features_batch(windows, fs=eng.fs, rpms=float(eng.rpms[i]))
    errors = model.errors(X)
    emb = model.embed(X)
    mean_err = float(np.mean(errors))
    return jsonify(
        id=sample_id,
        label=_label(eng.conditions[i]),
        condition=eng.conditions[i],
        threshold=model.threshold,
        mean_error=mean_err,
        over_threshold=bool(mean_err > model.threshold),
        errors=[float(e) for e in errors],
        embedding=[{"x": float(p[0]), "y": float(p[1])} for p in emb],
    )
