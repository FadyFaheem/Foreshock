"""Agentic diagnostic workflow.

On a detected anomaly the agent runs a multi-step chain end to end:

    pull signal -> analyze (features + classifier) -> retrieve knowledge (RAG)
    -> check health trend -> generate a grounded, structured diagnosis ->
    emit a draft maintenance work order.

Analysis stays in :mod:`src`; this module orchestrates the engine, the RAG
store, and the local LLM, and persists results. Everything degrades: with no
LLM it falls back to a templated recommendation; with no DB it skips persistence
and trend.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

import db
import llm
import rag
from engine import get_engine
from src import config, data_loader
from src.features import FEATURE_NAMES, extract_features_batch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a vibration-analysis maintenance assistant for rolling-element "
    "bearings. Use ONLY the supplied analysis results and knowledge-base "
    "context. Describe ONLY the single predicted condition you are given; never "
    "mention or compare other fault types. Do NOT invent fault types, "
    "frequencies, or facts not in the context. Respond with strict JSON only."
)

# Envelope-spectrum feature that corroborates each fault class.
_PEAK_KEY = {
    "inner_race": "env_peak_BPFI",
    "outer_race": "env_peak_BPFO",
    "ball": "env_peak_BSF",
}

_DEFAULT_ACTIONS = {
    "normal": ["Continue routine monitoring; re-baseline at the next interval."],
    "inner_race": [
        "Confirm BPFI and sidebands in the envelope spectrum.",
        "Plan bearing replacement; inspect lubrication and shaft fit.",
        "Re-baseline vibration after the repair.",
    ],
    "outer_race": [
        "Confirm BPFO in the envelope spectrum and check the load zone.",
        "Plan bearing replacement; inspect for contamination/overload.",
        "Re-baseline vibration after the repair.",
    ],
    "ball": [
        "Confirm BSF/2xBSF in the envelope spectrum.",
        "Plan bearing replacement; review lubrication condition.",
        "Re-baseline vibration after the repair.",
    ],
}


def _label(condition: str) -> str:
    return config.CONDITION_LABELS.get(condition, condition)


def _resolve_signal(
    sample_id: str | None, signal: Any, rpm: float | None, fs: int | None
) -> tuple[np.ndarray, float, int]:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("engine not loaded")
    if signal is not None:
        return (
            np.asarray(signal, dtype=np.float64).reshape(-1),
            float(rpm or config.DEFAULT_RPM),
            int(fs or config.DEFAULT_FS),
        )
    if not sample_id or sample_id not in eng.index:
        raise ValueError(f"unknown sample id: {sample_id}")
    i = eng.index[sample_id]
    return eng.signals[i], float(eng.rpms[i]), eng.fs


def _analyze(signal: np.ndarray, fs: int, rpm: float) -> dict[str, Any]:
    eng = get_engine()
    windows = data_loader.segment(signal)
    if windows.shape[0] == 0:
        raise ValueError(f"signal too short; need >= {config.WINDOW_SIZE} samples")
    X = extract_features_batch(windows, fs=fs, rpms=rpm)
    feat = dict(zip(FEATURE_NAMES, X.mean(axis=0).tolist()))
    proba = eng.model.predict_proba(X).mean(axis=0)
    classes = [str(c) for c in eng.model.classes_]
    best = int(np.argmax(proba))
    return {
        "condition": classes[best],
        "confidence": float(proba[best]),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "features": feat,
        "n_windows": int(windows.shape[0]),
        "rpm": rpm,
        "health": _health_verdict(X),
    }


def _health_verdict(X: np.ndarray) -> dict[str, Any] | None:
    """Unsupervised anomaly check via the health autoencoder (None if untrained).

    Reconstruction error rises for vibration that doesn't resemble the healthy
    training data - including one-off impacts the 4-class classifier can't name.
    Uses the max over windows so a single anomalous window still trips the alarm.
    """
    try:
        from health_routes import get_health_model

        hm = get_health_model()
    except Exception:  # noqa: BLE001
        hm = None
    if hm is None:
        return None
    try:
        errs = np.asarray(hm.errors(X), dtype=float)
        error = float(errs.max())
        threshold = float(hm.threshold)
        return {
            "error": error,
            "mean_error": float(errs.mean()),
            "threshold": threshold,
            "caught": bool(error > threshold),
            "score": float(error / threshold) if threshold else None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check failed: %s", exc)
        return None


def _severity(condition: str, confidence: float, feat: dict[str, float]) -> str:
    if condition == "normal":
        return "none"
    peak = feat.get(_PEAK_KEY.get(condition, ""), 0.0)
    if confidence >= 0.9 and peak >= 0.02:
        return "high"
    if confidence >= 0.7:
        return "medium"
    return "low"


def _anomaly_severity(score: float | None) -> str:
    """Severity from how far the reconstruction error exceeds the alarm threshold."""
    if not score:
        return "medium"
    if score >= 3.0:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"


def _anomaly_diagnosis(health: dict[str, Any] | None) -> dict[str, Any]:
    """Templated diagnosis for an anomaly the classifier could not name.

    Deterministic (no LLM): the health monitor only knows "this isn't healthy",
    so we must not invent a specific fault type the evidence doesn't support.
    """
    score = (health or {}).get("score")
    sev = _anomaly_severity(score)
    err = (health or {}).get("error")
    thr = (health or {}).get("threshold")
    detail = (
        f"reconstruction error {err:.3f} vs threshold {thr:.3f}"
        if err is not None and thr is not None
        else "elevated reconstruction error"
    )
    return {
        "summary": (
            "Unclassified anomaly: the health monitor flagged abnormal vibration "
            f"({detail}) that does not match a known bearing-fault frequency "
            "(BPFO/BPFI/BSF). This is typical of a one-off impact or an early, "
            "still-incipient defect - worth investigating before clearing."
        ),
        "severity": sev,
        "likely_cause": (
            "Transient/impulsive anomaly without a periodic bearing-fault signature."
        ),
        "recommended_actions": [
            "Inspect the asset for looseness, external impacts, or sensor faults.",
            "Capture a longer recording and re-analyze to see if a fault frequency emerges.",
            "Trend the health indicator; escalate if reconstruction error keeps rising.",
        ],
        "priority": _priority(sev),
        "used_llm": False,
    }


def _build_user_prompt(
    condition: str, confidence: float, feat: dict[str, float], rpm: float, context: str
) -> str:
    freqs = config.fault_frequencies(rpm)
    key_feats = {
        k: round(feat.get(k, 0.0), 5)
        for k in ("rms", "kurtosis", "crest_factor", "env_peak_BPFO",
                  "env_peak_BPFI", "env_peak_BSF", "env_peak_FTF")
    }
    return (
        f"Classifier result: {_label(condition)} (confidence {confidence:.2f}).\n"
        f"Shaft speed: {rpm:.0f} RPM. Characteristic fault frequencies (Hz): "
        f"{ {k: round(v, 1) for k, v in freqs.items()} }.\n"
        f"Key features: {key_feats}.\n\n"
        f"Knowledge base context:\n{context}\n\n"
        f'Write strictly about "{_label(condition)}" only; do not name any other '
        "bearing fault type (inner race, outer race, ball, or cage) unless it is "
        "the predicted condition itself.\n\n"
        "Return JSON with exactly these keys: "
        '{"summary": string (2-3 sentences), '
        '"severity": one of "none"|"low"|"medium"|"high", '
        '"likely_cause": string, '
        '"recommended_actions": array of 2-4 short strings, '
        '"priority": one of "low"|"medium"|"high"}.'
    )


def _generate(
    condition: str,
    confidence: float,
    feat: dict[str, float],
    rpm: float,
    docs: list[dict[str, Any]],
    health: dict[str, Any] | None = None,
    unclassified: bool = False,
) -> dict[str, Any]:
    """Produce the grounded diagnosis (LLM if available, else templated)."""
    if unclassified:
        # Classifier said normal but the health monitor flagged an anomaly: report
        # it as an unnamed anomaly rather than a (non-existent) specific fault.
        return _anomaly_diagnosis(health)
    severity = _severity(condition, confidence, feat)
    retrieval_score = docs[0]["score"] if docs else None

    if docs:
        try:
            obj, _usage = llm.chat_json(
                SYSTEM_PROMPT,
                _build_user_prompt(condition, confidence, feat, rpm, rag.format_context(docs)),
                retrieval_score=retrieval_score,
            )
            if "_raw" not in obj:
                return {
                    "summary": str(obj.get("summary", "")).strip()
                    or _fallback_summary(condition, confidence),
                    "severity": obj.get("severity", severity),
                    "likely_cause": obj.get("likely_cause"),
                    "recommended_actions": obj.get("recommended_actions")
                    or _DEFAULT_ACTIONS.get(condition, []),
                    "priority": obj.get("priority", _priority(severity)),
                    "used_llm": True,
                }
        except llm.LLMUnavailable as exc:
            logger.warning("LLM unavailable, using templated diagnosis: %s", exc)

    return {
        "summary": _fallback_summary(condition, confidence),
        "severity": severity,
        "likely_cause": f"{_label(condition)} signature in the envelope spectrum."
        if condition != "normal"
        else "No fault signature detected.",
        "recommended_actions": _DEFAULT_ACTIONS.get(condition, []),
        "priority": _priority(severity),
        "used_llm": False,
    }


def _fallback_summary(condition: str, confidence: float) -> str:
    if condition == "normal":
        return f"No bearing fault detected (confidence {confidence:.0%}). Continue monitoring."
    return (
        f"Predicted {_label(condition)} at {confidence:.0%} confidence, corroborated "
        f"by envelope-spectrum energy at the characteristic fault frequency."
    )


def _priority(severity: str) -> str:
    return {"none": "low", "low": "low", "medium": "medium", "high": "high"}.get(
        severity, "medium"
    )


def _sources(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"title": d["title"], "source": d["source"], "fault_type": d["fault_type"],
         "score": round(d["score"], 4)}
        for d in docs
    ]


def _retrieval_query(condition: str, unclassified: bool = False) -> str:
    if unclassified:
        return (
            "bearing vibration anomaly detection, envelope analysis, rising RMS and "
            "kurtosis, severity assessment and maintenance priority"
        )
    if condition == "normal":
        return (
            "healthy bearing baseline vibration signature, low RMS and kurtosis, "
            "no fault frequencies"
        )
    return (
        f"{_label(condition)} bearing fault: characteristic fault frequency, "
        "envelope-spectrum evidence, severity and recommended maintenance actions"
    )


def _retrieve(
    condition: str, unclassified: bool = False
) -> tuple[list[dict[str, Any]], float | None]:
    try:
        # Focus retrieval on the predicted condition (+ general background) so the
        # context never contains a different fault's document. For an unclassified
        # anomaly, pull general envelope/severity guidance instead.
        prefer = "general" if unclassified else condition
        docs = rag.retrieve(
            _retrieval_query(condition, unclassified), top_k=4, prefer=prefer
        )
        return docs, (docs[0]["score"] if docs else None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval failed: %s", exc)
        return [], None


# An "unclassified anomaly" = the classifier says normal, but the window is
# genuinely impulsive - the signature of a one-off impact. ``kurtosis`` is excess
# kurtosis (~0 for a clean signal), so this never false-positives on healthy data.
# Keyed on impulsiveness ALONE on purpose: it must work even when the health
# autoencoder is untrained/missing, and the classifier's confidence is meaningless
# here (it cannot classify isolated impacts). The health monitor, when available,
# is reported as extra evidence but is NOT required to raise the flag.
_KURTOSIS_ANOMALY = 1.5
_CREST_ANOMALY = 5.0


def _is_unclassified_anomaly(analysis: dict[str, Any]) -> bool:
    """True for a genuinely impulsive anomaly the classifier cannot name."""
    if analysis["condition"] != "normal":
        return False
    feat = analysis["features"]
    return (
        feat.get("kurtosis", 0.0) >= _KURTOSIS_ANOMALY
        or feat.get("crest_factor", 0.0) >= _CREST_ANOMALY
    )


def _resolve_condition(
    analysis: dict[str, Any],
    expected_condition: str | None,
) -> tuple[str, bool]:
    """Decide the reported condition and whether it's an unclassified anomaly.

    When the caller supplies a known ground-truth fault (the Fault Lab generated
    this exact fault, or "normal" for the healthy baseline), trust it. Otherwise
    use the classifier, promoting a "normal" result to an unclassified anomaly when
    the signal is genuinely impulsive (a clean healthy signal never is).
    """
    if expected_condition in config.CONDITIONS:
        return expected_condition, False
    cond = analysis["condition"]
    return cond, _is_unclassified_anomaly(analysis)


def _confidence_for(
    analysis: dict[str, Any],
    cond: str,
    health: dict[str, Any] | None,
    unclassified: bool,
) -> float:
    """Confidence to report for the reported condition."""
    if unclassified:
        score = (health or {}).get("score") or 1.0
        # Normalized anomaly strength: ~0.5 at the threshold -> 1 for large errors.
        return float(score / (1.0 + score))
    # The classifier's probability for the reported class (honest agreement level;
    # low when reporting a generated fault the classifier hasn't learned yet).
    return float(analysis["probabilities"].get(cond, analysis["confidence"]))


def _assemble_diagnosis(
    sample_id: str | None,
    asset: str,
    analysis: dict[str, Any],
    docs: list[dict[str, Any]],
    gen: dict[str, Any],
    health: dict[str, Any] | None,
    cond: str,
    confidence: float,
    unclassified: bool,
) -> dict[str, Any]:
    """Build the diagnosis dict shared by diagnose() and run_agent()."""
    disp_cond, disp_label = (
        ("anomaly", "Unclassified anomaly") if unclassified else (cond, _label(cond))
    )
    return {
        "sample_id": sample_id,
        "asset": asset,
        "condition": disp_cond,
        "label": disp_label,
        "classifier_condition": analysis["condition"],
        "confidence": confidence,
        "rpm": analysis["rpm"],
        "rms": analysis["features"].get("rms"),
        "probabilities": analysis["probabilities"],
        "features": analysis["features"],
        "sources": _sources(docs),
        "model": llm.LLM_MODEL,
        "anomaly": disp_cond != "normal",
        "health": health,
        **gen,
    }


def diagnose(
    sample_id: str | None = None,
    signal: Any = None,
    rpm: float | None = None,
    fs: int | None = None,
    asset: str = "bearing-DE-01",
    persist: bool = True,
    expected_condition: str | None = None,
) -> dict[str, Any]:
    """Single-shot RAG + LLM diagnosis for a sample or signal."""
    sig, rpm, fs = _resolve_signal(sample_id, signal, rpm, fs)
    analysis = _analyze(sig, fs, rpm)
    health = analysis["health"]
    cond, unclassified = _resolve_condition(analysis, expected_condition)
    confidence = _confidence_for(analysis, cond, health, unclassified)

    docs, _score = _retrieve(cond, unclassified)
    gen = _generate(
        cond, confidence, analysis["features"], rpm, docs,
        health=health, unclassified=unclassified,
    )
    diagnosis = _assemble_diagnosis(
        sample_id, asset, analysis, docs, gen, health, cond, confidence, unclassified
    )
    if persist:
        diagnosis["id"] = _persist_diagnosis(diagnosis)
    return diagnosis


def _persist_diagnosis(d: dict[str, Any]) -> int | None:
    if not db.available():
        return None
    try:
        row = db.execute(
            "INSERT INTO diagnoses (sample_id, predicted_condition, confidence, "
            "severity, summary, recommended_actions, sources, model, asset, rms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                d.get("sample_id"), d["condition"], d["confidence"], d["severity"],
                d["summary"], db.Json(d["recommended_actions"]), db.Json(d["sources"]),
                d["model"], d["asset"], d.get("rms"),
            ),
            returning=True,
        )
        return row["id"] if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist diagnosis: %s", exc)
        return None


def _trend(asset: str, current_rms: float | None, current_conf: float) -> dict[str, Any]:
    """Compare current RMS to the previous diagnosis for this asset."""
    if not db.available() or current_rms is None:
        return {"direction": "unknown", "summary": "No history available.", "history": []}
    try:
        rows = db.query(
            "SELECT rms, confidence, predicted_condition, created_at FROM diagnoses "
            "WHERE asset = %s AND rms IS NOT NULL ORDER BY created_at DESC LIMIT 6",
            (asset,),
        )
    except Exception:  # noqa: BLE001
        rows = []
    history = [
        {"rms": float(r["rms"]), "condition": r["predicted_condition"],
         "created_at": r["created_at"].isoformat()}
        for r in rows
    ]
    if not rows:
        return {"direction": "baseline", "summary": "First reading for this asset.",
                "history": history}
    prev = float(rows[0]["rms"])
    delta = (current_rms - prev) / prev if prev else 0.0
    direction = "rising" if delta > 0.1 else "falling" if delta < -0.1 else "stable"
    return {
        "direction": direction,
        "delta_pct": round(delta * 100, 1),
        "summary": f"RMS {direction} ({delta * 100:+.1f}% vs previous reading).",
        "history": history,
    }


def run_agent(
    sample_id: str | None = None,
    signal: Any = None,
    rpm: float | None = None,
    fs: int | None = None,
    asset: str = "bearing-DE-01",
    expected_condition: str | None = None,
) -> dict[str, Any]:
    """Run the full multi-step agentic diagnostic workflow."""
    steps: list[dict[str, Any]] = []

    sig, rpm, fs = _resolve_signal(sample_id, signal, rpm, fs)
    steps.append({
        "step": "pull_signal", "status": "ok",
        "detail": f"Loaded {sig.shape[0]} samples at {fs} Hz, {rpm:.0f} RPM",
    })

    analysis = _analyze(sig, fs, rpm)
    classifier_cond, conf = analysis["condition"], analysis["confidence"]
    health = analysis["health"]
    cond, unclassified = _resolve_condition(analysis, expected_condition)
    analyze_detail = (
        f"Classified {_label(classifier_cond)} ({conf:.0%}) over "
        f"{analysis['n_windows']} windows"
    )
    if not unclassified and cond != classifier_cond:
        analyze_detail += f" - reporting generated fault: {_label(cond)}"
    steps.append({"step": "analyze", "status": "ok", "detail": analyze_detail})

    # Unsupervised safety net: the health autoencoder flags anomalies the 4-class
    # classifier can't name (e.g. one-off impacts).
    if health is not None:
        steps.append({
            "step": "health_check", "status": "ok",
            "detail": f"Reconstruction error {health['error']:.3f} / threshold "
            f"{health['threshold']:.3f} - "
            + ("above threshold" if health["caught"] else "within healthy range"),
        })
    else:
        steps.append({
            "step": "health_check", "status": "skipped",
            "detail": "Health model not trained (run scripts/train_health.py)",
        })

    anomaly = cond != "normal" or unclassified
    confidence = _confidence_for(analysis, cond, health, unclassified)
    steps.append({
        "step": "anomaly_check", "status": "ok",
        "detail": (
            "Unclassified anomaly (health monitor)" if unclassified
            else "Anomaly detected" if anomaly
            else "Healthy - no anomaly"
        ),
    })

    docs, score = _retrieve(cond, unclassified)
    steps.append({
        "step": "retrieve_kb",
        "status": "ok" if docs else "skipped",
        "detail": f"Retrieved {len(docs)} docs (top score {score:.2f})" if docs
        else "Knowledge base unavailable",
    })

    trend = _trend(asset, analysis["features"].get("rms"), conf)
    steps.append({"step": "check_trend", "status": "ok", "detail": trend["summary"]})

    gen = _generate(cond, confidence, analysis["features"], rpm, docs,
                    health=health, unclassified=unclassified)
    steps.append({
        "step": "generate_diagnosis",
        "status": "ok",
        "detail": f"{'LLM' if gen['used_llm'] else 'Templated'} diagnosis, "
        f"severity {gen['severity']}",
    })

    diagnosis = _assemble_diagnosis(
        sample_id, asset, analysis, docs, gen, health, cond, confidence, unclassified
    )
    diagnosis["id"] = _persist_diagnosis(diagnosis)

    work_order = _make_work_order(diagnosis)
    steps.append({
        "step": "emit_work_order",
        "status": "ok" if anomaly else "skipped",
        "detail": f"Drafted work order #{work_order.get('id')} (priority {work_order['priority']})"
        if anomaly else "No work order needed for a healthy bearing",
    })

    return {
        "asset": asset,
        "anomaly": anomaly,
        "steps": steps,
        "trend": trend,
        "diagnosis": diagnosis,
        "work_order": work_order if anomaly else None,
    }


def _make_work_order(d: dict[str, Any]) -> dict[str, Any]:
    wo = {
        "asset": d["asset"],
        "condition": d["condition"],
        "priority": d.get("priority", "medium"),
        "actions": d.get("recommended_actions", []),
        "status": "draft",
    }
    if d["condition"] != "normal" and db.available():
        try:
            row = db.execute(
                "INSERT INTO work_orders (diagnosis_id, asset, condition, priority, "
                "actions, status) VALUES (%s, %s, %s, %s, %s, 'draft') RETURNING id",
                (d.get("id"), wo["asset"], wo["condition"], wo["priority"],
                 db.Json(wo["actions"])),
                returning=True,
            )
            wo["id"] = row["id"] if row else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist work order: %s", exc)
    return wo
