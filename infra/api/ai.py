"""AI blueprint: RAG diagnosis, agentic workflow, evals, and observability.

Thin HTTP layer over :mod:`agent`, :mod:`evals`, :mod:`rag`, :mod:`llm`, and
:mod:`db`. All endpoints degrade gracefully when the LLM or database is down.
"""

from __future__ import annotations

import logging

import numpy as np
from flask import Blueprint, jsonify, request

import agent
import db
import evals
import llm
import rag
from engine import get_engine
from predict import _parse_upload
from src import config, synthetic
from src.features import downsample, envelope_spectrum_for_display

logger = logging.getLogger(__name__)

ai_bp = Blueprint("ai", __name__, url_prefix="/api")


def _resolve_request_signal():
    """Extract (sample_id, signal, rpm, fs) from a diagnose/agent request."""
    file = request.files.get("file")
    body = request.get_json(silent=True) or {}
    sample_id = request.form.get("sample_id") or body.get("sample_id")
    asset = request.form.get("asset") or body.get("asset") or "bearing-DE-01"

    if file is not None and file.filename:
        signal, rpm = _parse_upload(file.filename, file.read())
        return {"signal": signal, "rpm": rpm, "fs": config.DEFAULT_FS, "asset": asset}
    if sample_id:
        return {"sample_id": sample_id, "asset": asset}
    return None


@ai_bp.get("/ai/status")
def ai_status():
    """Report availability of the AI subsystems (for the frontend)."""
    return jsonify(
        db=db.available(),
        llm=llm.available(),
        kb_size=rag.kb_count(),
        model=llm.LLM_MODEL,
        embed_model=llm.EMBED_MODEL,
    )


@ai_bp.post("/diagnose")
def diagnose():
    """RAG + LLM diagnosis for a sample id or uploaded signal."""
    if get_engine() is None:
        return jsonify(error="Model not loaded. Run scripts/train.py."), 503
    try:
        args = _resolve_request_signal()
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Could not parse input: {exc}"), 400
    if args is None:
        return jsonify(error="Provide a sample_id or upload a file"), 400
    try:
        return jsonify(agent.diagnose(**args))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@ai_bp.post("/agent")
def run_agent():
    """Run the full agentic diagnostic workflow."""
    if get_engine() is None:
        return jsonify(error="Model not loaded. Run scripts/train.py."), 503
    try:
        args = _resolve_request_signal()
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Could not parse input: {exc}"), 400
    if args is None:
        return jsonify(error="Provide a sample_id or upload a file"), 400
    try:
        return jsonify(agent.run_agent(**args))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


# Condition -> the characteristic envelope frequency that evidences it, so the UI
# can point out exactly where the fault shows up in the spectrum.
_FAULT_FREQ_KEY = {
    "inner_race": "BPFI",
    "outer_race": "BPFO",
    "ball": "BSF",
}


@ai_bp.post("/inject/diagnose")
def inject_diagnose():
    """Generate a fault, then run the agentic RAG + LLM analysis on it.

    Body: ``{signal: [...], points: [idx...], amplitude, fs, rpm, asset}``. Injects
    damped impulses into a healthy window at the chosen spots and runs the full
    agent, so the UI can show the synthesized waveform, the envelope spectrum with
    the characteristic fault frequency marked, and the grounded diagnosis.
    """
    if get_engine() is None:
        return jsonify(error="Model not loaded. Run scripts/train.py."), 503
    body = request.get_json(silent=True) or {}
    sig = np.asarray(body.get("signal") or [], dtype=np.float64).reshape(-1)
    points = [int(p) for p in (body.get("points") or [])]
    amplitude = float(body.get("amplitude", 1.0))
    if sig.shape[0] < config.WINDOW_SIZE:
        return jsonify(error=f"signal must have >= {config.WINDOW_SIZE} samples"), 400
    fs = int(body.get("fs", config.DEFAULT_FS))
    rpm = float(body.get("rpm", config.DEFAULT_RPM))
    asset = body.get("asset") or "fault-lab-01"

    modified = synthetic.inject_impulses(sig, points, amplitude=amplitude, fs=fs)
    try:
        run = agent.run_agent(signal=modified, rpm=rpm, fs=fs, asset=asset)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    t = np.arange(modified.shape[0]) / fs
    f_env, m_env = envelope_spectrum_for_display(modified, fs)
    freqs = config.fault_frequencies(rpm)
    key = _FAULT_FREQ_KEY.get(run["diagnosis"]["condition"])
    return jsonify(
        agent=run,
        injected_points=points,
        amplitude=amplitude,
        # Only the detected fault's frequency, so the UI marks the one the AI saw.
        marked_frequency={key: round(freqs[key], 1)} if key else {},
        fault_frequencies={k: round(v, 1) for k, v in freqs.items()},
        waveform={
            "t": downsample(t, config.WAVEFORM_POINTS).tolist(),
            "x": downsample(modified, config.WAVEFORM_POINTS).tolist(),
        },
        envelope={"f": f_env.tolist(), "mag": m_env.tolist()},
    )


@ai_bp.get("/work_orders")
def work_orders():
    if not db.available():
        return jsonify(error="Database unavailable"), 503
    rows = db.query(
        "SELECT id, diagnosis_id, asset, condition, priority, actions, status, "
        "created_at FROM work_orders ORDER BY created_at DESC LIMIT 25"
    )
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return jsonify(rows)


@ai_bp.get("/evals")
def latest_evals():
    """Return the most recent persisted eval run (or empty)."""
    if not db.available():
        return jsonify(error="Database unavailable"), 503
    row = db.query(
        "SELECT id, suite, total, passed, diagnosis_accuracy, retrieval_precision, "
        "retrieval_recall, hallucination_rate, details, created_at "
        "FROM eval_runs ORDER BY created_at DESC LIMIT 1",
        fetch_one=True,
    )
    if not row:
        return jsonify({})
    row["created_at"] = row["created_at"].isoformat()
    return jsonify(row)


@ai_bp.post("/evals/run")
def run_evals_endpoint():
    if get_engine() is None:
        return jsonify(error="Model not loaded. Run scripts/train.py."), 503
    return jsonify(evals.run_evals())


@ai_bp.get("/observability")
def observability():
    """Aggregate LLM/embedding call metrics for the AI observability panel."""
    if not db.available():
        return jsonify(error="Database unavailable"), 503

    summary = db.query(
        "SELECT COUNT(*) AS calls, "
        "COALESCE(AVG(latency_ms), 0) AS avg_latency_ms, "
        "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
        "COALESCE(AVG(retrieval_score) FILTER (WHERE retrieval_score IS NOT NULL), 0) "
        "AS avg_retrieval_score, "
        "COALESCE(SUM(CASE WHEN ok THEN 0 ELSE 1 END), 0) AS errors "
        "FROM llm_calls",
        fetch_one=True,
    )
    by_op = db.query(
        "SELECT operation, COUNT(*) AS calls, "
        "COALESCE(AVG(latency_ms), 0) AS avg_latency_ms, "
        "COALESCE(SUM(total_tokens), 0) AS total_tokens "
        "FROM llm_calls GROUP BY operation ORDER BY operation"
    )
    recent = db.query(
        "SELECT operation, model, latency_ms, prompt_tokens, completion_tokens, "
        "total_tokens, retrieval_score, ok, created_at "
        "FROM llm_calls ORDER BY created_at DESC LIMIT 25"
    )
    for r in recent:
        r["created_at"] = r["created_at"].isoformat()

    return jsonify(summary=summary, by_operation=by_op, recent=recent)
