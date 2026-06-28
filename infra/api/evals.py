"""Eval harness for the LLM / RAG diagnostic layer.

Runs a small suite of fault scenarios with known ground truth and reports:

- diagnosis accuracy (predicted condition vs expected),
- retrieval precision (fraction of retrieved docs that are on-topic, i.e. the
  expected fault or general background, not a different fault),
- retrieval recall (did the expected fault's document surface in top-k),
- hallucination rate (did the generated summary assert a *different* fault than
  the expected one).

Runs offline-friendly: retrieval/accuracy work without the LLM; the
hallucination check is skipped per-case if generation was unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import agent
import db
import engine
import rag

logger = logging.getLogger(__name__)

# Ground-truth scenarios mapped to the bundled demo samples.
TEST_CASES: list[dict[str, str]] = [
    {"sample_id": "normal", "expected": "normal"},
    {"sample_id": "inner_race", "expected": "inner_race"},
    {"sample_id": "outer_race", "expected": "outer_race"},
    {"sample_id": "ball", "expected": "ball"},
]

# Keywords that indicate a given fault is being asserted in free text.
_FAULT_TERMS = {
    "inner_race": ["inner race", "bpfi"],
    "outer_race": ["outer race", "bpfo"],
    "ball": ["ball spin", "rolling element", "bsf"],
    "normal": ["healthy", "no fault", "no bearing fault", "normal"],
}


def _hallucinated(summary: str, expected: str) -> bool:
    """True if the summary asserts a fault other than the expected one."""
    text = (summary or "").lower()
    for cond, terms in _FAULT_TERMS.items():
        if cond == expected:
            continue
        if cond == "normal":
            continue  # mentioning "normal" is not a fault assertion
        if any(t in text for t in terms):
            return True
    return False


def _retrieval_metrics(
    sources: list[dict[str, Any]], expected: str
) -> tuple[float, float]:
    """Return (precision, recall) for one case's retrieved sources."""
    if not sources:
        return 0.0, 0.0
    on_topic = [s for s in sources if s["fault_type"] in (expected, "general")]
    precision = len(on_topic) / len(sources)
    recall = 1.0 if any(s["fault_type"] == expected for s in sources) else 0.0
    return precision, recall


def run_evals(top_k: int = 4, persist: bool = True) -> dict[str, Any]:
    """Run the suite and return a metrics report (also persisted to eval_runs)."""
    if engine.get_engine() is None:
        engine.load_engine()  # standalone (CLI) use; the API loads at startup

    details: list[dict[str, Any]] = []
    kb = rag.kb_count()

    for case in TEST_CASES:
        expected = case["expected"]
        try:
            d = agent.diagnose(sample_id=case["sample_id"], persist=False)
        except Exception as exc:  # noqa: BLE001
            details.append({**case, "error": str(exc), "correct": False})
            continue

        precision, recall = _retrieval_metrics(d["sources"], expected)
        correct = d["condition"] == expected
        halluc = _hallucinated(d.get("summary", ""), expected) if d.get("used_llm") else False
        details.append({
            "sample_id": case["sample_id"],
            "expected": expected,
            "predicted": d["condition"],
            "confidence": round(d["confidence"], 4),
            "correct": correct,
            "retrieval_precision": round(precision, 3),
            "retrieval_recall": round(recall, 3),
            "hallucinated": halluc,
            "used_llm": d.get("used_llm", False),
            "passed": correct and not halluc,
            "top_source": d["sources"][0]["title"] if d["sources"] else None,
        })

    total = len(details)
    scored = [x for x in details if "error" not in x]
    n = len(scored) or 1
    report = {
        "suite": "fault-scenarios",
        "total": total,
        "passed": sum(1 for x in scored if x.get("passed")),
        "diagnosis_accuracy": round(sum(1 for x in scored if x["correct"]) / n, 3),
        "retrieval_precision": round(sum(x["retrieval_precision"] for x in scored) / n, 3),
        "retrieval_recall": round(sum(x["retrieval_recall"] for x in scored) / n, 3),
        "hallucination_rate": round(sum(1 for x in scored if x.get("hallucinated")) / n, 3),
        "kb_size": kb,
        "details": details,
    }

    if persist and db.available():
        try:
            row = db.execute(
                "INSERT INTO eval_runs (suite, total, passed, diagnosis_accuracy, "
                "retrieval_precision, retrieval_recall, hallucination_rate, details) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id, created_at",
                (
                    report["suite"], report["total"], report["passed"],
                    report["diagnosis_accuracy"], report["retrieval_precision"],
                    report["retrieval_recall"], report["hallucination_rate"],
                    db.Json(details),
                ),
                returning=True,
            )
            if row:
                report["id"] = row["id"]
                report["created_at"] = row["created_at"].isoformat()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist eval run: %s", exc)
    return report
