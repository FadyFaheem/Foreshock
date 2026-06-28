"""Foreshock API blueprint.

Domain routes for the demo: list samples, fetch a signal's waveform/spectra,
and predict a bearing condition. Routes are thin -- they validate input, call
into :mod:`src` and :mod:`engine`, and format JSON. No analysis here.
"""

from __future__ import annotations

import numpy as np
from flask import Blueprint, jsonify, request

from engine import get_engine
from health_routes import get_health_model
from src import config, data_loader, synthetic
from src.features import (
    FEATURE_NAMES,
    downsample,
    envelope_spectrum_for_display,
    extract_features_batch,
    fft_spectrum_for_display,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _label(condition: str) -> str:
    return config.CONDITION_LABELS.get(condition, condition)


def _engine_or_503():
    """Return (engine, None) or (None, error_response) if not loaded."""
    eng = get_engine()
    if eng is None:
        resp = (
            jsonify(error="Model not loaded. Run `python scripts/train.py` then restart."),
            503,
        )
        return None, resp
    return eng, None


@api_bp.get("/samples")
def list_samples():
    """List the built-in sample signals (one per condition)."""
    eng, err = _engine_or_503()
    if err:
        return err
    return jsonify(
        [
            {"id": sid, "condition": c, "label": _label(c)}
            for sid, c in zip(eng.sample_ids, eng.conditions)
        ]
    )


@api_bp.get("/signal/<sample_id>")
def get_signal(sample_id: str):
    """Return waveform, FFT spectrum, and envelope spectrum for a sample."""
    eng, err = _engine_or_503()
    if err:
        return err
    if sample_id not in eng.index:
        return jsonify(error=f"Unknown sample id: {sample_id}"), 404

    i = eng.index[sample_id]
    signal = eng.signals[i]
    rpm = float(eng.rpms[i])

    t = np.arange(signal.shape[0]) / eng.fs
    f_fft, m_fft = fft_spectrum_for_display(signal, eng.fs)
    f_env, m_env = envelope_spectrum_for_display(signal, eng.fs)

    return jsonify(
        id=sample_id,
        condition=eng.conditions[i],
        label=_label(eng.conditions[i]),
        fs=eng.fs,
        rpm=rpm,
        fault_frequencies=config.fault_frequencies(rpm),
        waveform={
            "t": downsample(t, config.WAVEFORM_POINTS).tolist(),
            "x": downsample(signal, config.WAVEFORM_POINTS).tolist(),
        },
        spectrum={"f": f_fft.tolist(), "mag": m_fft.tolist()},
        envelope={"f": f_env.tolist(), "mag": m_env.tolist()},
    )


def _parse_upload(filename: str | None, raw: bytes) -> tuple[np.ndarray, float]:
    """Parse an uploaded .mat or .csv into ``(signal, rpm)`` via the engine."""
    name = (filename or "").lower()
    try:
        if name.endswith(".csv"):
            return data_loader.load_csv_bytes(raw)
        if name.endswith(".mat"):
            return data_loader.load_mat_bytes(raw)
        try:
            return data_loader.load_mat_bytes(raw)
        except Exception:
            return data_loader.load_csv_bytes(raw)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 to the client
        raise ValueError(str(exc)) from exc


@api_bp.post("/predict")
def predict():
    """Predict the bearing condition for a sample id or an uploaded file.

    Accepts ``multipart/form-data`` with either a ``sample_id`` field or a
    ``file`` upload (.mat or .csv). Class probabilities are averaged across all
    windows of the signal.
    """
    eng, err = _engine_or_503()
    if err:
        return err

    file = request.files.get("file")
    sample_id = request.form.get("sample_id")

    if file is not None and file.filename:
        try:
            signal, rpm = _parse_upload(file.filename, file.read())
        except ValueError as exc:
            return jsonify(error=f"Could not parse uploaded file: {exc}"), 400
        fs = config.DEFAULT_FS
    elif sample_id:
        if sample_id not in eng.index:
            return jsonify(error=f"Unknown sample id: {sample_id}"), 404
        i = eng.index[sample_id]
        signal, rpm, fs = eng.signals[i], float(eng.rpms[i]), eng.fs
    else:
        return jsonify(error="Provide a sample_id or upload a file"), 400

    windows = data_loader.segment(signal)
    if windows.shape[0] == 0:
        return (
            jsonify(error=f"Signal too short: need at least {config.WINDOW_SIZE} samples"),
            400,
        )

    X = extract_features_batch(windows, fs=fs, rpms=rpm)
    proba = eng.model.predict_proba(X).mean(axis=0)
    classes = [str(c) for c in eng.model.classes_]
    best = int(np.argmax(proba))

    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    probabilities = sorted(
        (
            {"condition": c, "label": _label(c), "probability": float(p)}
            for c, p in zip(classes, proba)
        ),
        key=lambda d: order.get(d["condition"], 99),
    )
    features = [
        {"name": n, "value": float(v)} for n, v in zip(FEATURE_NAMES, X.mean(axis=0))
    ]

    return jsonify(
        prediction=classes[best],
        prediction_label=_label(classes[best]),
        confidence=float(proba[best]),
        n_windows=int(windows.shape[0]),
        rpm=rpm,
        probabilities=probabilities,
        features=features,
    )


@api_bp.post("/random_test")
def random_test():
    """Generate a random labeled test window and classify it.

    Picks a random window from a (held-out) sample signal of the requested
    condition ("random" = any), optionally adds noise, and returns the true vs.
    predicted condition so the UI can show accuracy across trials.
    """
    eng, err = _engine_or_503()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    requested = request.form.get("condition") or body.get("condition") or "random"
    try:
        noise = float(request.form.get("noise") or body.get("noise") or 0.0)
    except (TypeError, ValueError):
        noise = 0.0

    conditions = eng.conditions
    rng = np.random.default_rng()
    if requested in conditions:
        idx = conditions.index(requested)
    else:
        idx = int(rng.integers(len(conditions)))

    actual = eng.conditions[idx]
    rpm = float(eng.rpms[idx])
    window = synthetic.add_noise(
        synthetic.random_window(eng.signals[idx], rng=rng), noise, rng=rng
    )

    X = extract_features_batch(window[np.newaxis, :], fs=eng.fs, rpms=rpm)
    proba = eng.model.predict_proba(X)[0]
    classes = [str(c) for c in eng.model.classes_]
    best = int(np.argmax(proba))
    pred = classes[best]

    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    probabilities = sorted(
        (
            {"condition": c, "label": _label(c), "probability": float(p)}
            for c, p in zip(classes, proba)
        ),
        key=lambda d: order.get(d["condition"], 99),
    )
    t = np.arange(window.shape[0]) / eng.fs

    return jsonify(
        actual=actual,
        actual_label=_label(actual),
        prediction=pred,
        prediction_label=_label(pred),
        correct=bool(pred == actual),
        confidence=float(proba[best]),
        probabilities=probabilities,
        rpm=rpm,
        noise=max(0.0, min(1.0, noise)),
        waveform={
            "t": downsample(t, config.WAVEFORM_POINTS).tolist(),
            "x": downsample(window, config.WAVEFORM_POINTS).tolist(),
        },
    )


@api_bp.get("/inject/base")
def inject_base():
    """Return a fresh healthy window for the fault-injection tool."""
    eng, err = _engine_or_503()
    if err:
        return err
    i = eng.index.get("normal", 0)
    window = synthetic.random_window(eng.signals[i], rng=np.random.default_rng())
    return jsonify(signal=window.tolist(), fs=eng.fs, rpm=float(eng.rpms[i]))


@api_bp.post("/inject")
def inject():
    """Inject defect impulses at chosen spots and report whether it's caught.

    Body: ``{signal: [...], points: [idx...], amplitude, fs, rpm}``. Runs the
    classifier and (if trained) the health autoencoder; ``caught`` is true if
    either flags the modified signal as non-healthy.
    """
    eng, err = _engine_or_503()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    sig = np.asarray(body.get("signal") or [], dtype=np.float64)
    points = body.get("points") or []
    amplitude = float(body.get("amplitude", 1.0))
    if sig.shape[0] < config.WINDOW_SIZE:
        return jsonify(error=f"signal must have >= {config.WINDOW_SIZE} samples"), 400
    fs = int(body.get("fs", eng.fs))
    rpm = float(body.get("rpm", config.DEFAULT_RPM))

    modified = synthetic.inject_impulses(sig, points, amplitude=amplitude, fs=fs)
    X = extract_features_batch(modified[np.newaxis, :], fs=fs, rpms=rpm)
    proba = eng.model.predict_proba(X)[0]
    classes = [str(c) for c in eng.model.classes_]
    best = int(np.argmax(proba))
    pred = classes[best]
    classifier_caught = pred != "normal"

    health = None
    hm = get_health_model()
    if hm is not None:
        err_val = float(hm.errors(X)[0])
        health = {
            "error": err_val,
            "threshold": float(hm.threshold),
            "caught": bool(err_val > hm.threshold),
        }

    caught = classifier_caught or bool(health and health["caught"])
    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    probabilities = sorted(
        (
            {"condition": c, "label": _label(c), "probability": float(p)}
            for c, p in zip(classes, proba)
        ),
        key=lambda d: order.get(d["condition"], 99),
    )
    t = np.arange(modified.shape[0]) / fs
    return jsonify(
        prediction=pred,
        prediction_label=_label(pred),
        confidence=float(proba[best]),
        classifier_caught=classifier_caught,
        caught=caught,
        n_points=len(points),
        amplitude=amplitude,
        probabilities=probabilities,
        health=health,
        waveform={
            "t": downsample(t, config.WAVEFORM_POINTS).tolist(),
            "x": downsample(modified, config.WAVEFORM_POINTS).tolist(),
        },
    )
