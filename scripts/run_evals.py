"""Run the LLM/RAG eval suite and print a report.

    python scripts/run_evals.py

Requires a trained model, a running database (seeded knowledge base), and the
local LLM. Persists the run to the eval_runs table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "infra" / "api"
for _p in (str(PROJECT_ROOT), str(API_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evals  # noqa: E402


def main() -> int:
    report = evals.run_evals()
    summary = {k: v for k, v in report.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    print("\nPer-case:")
    for d in report["details"]:
        flag = "PASS" if d.get("passed") else "FAIL"
        print(
            f"  [{flag}] {d.get('sample_id'):11s} "
            f"expected={d.get('expected')} predicted={d.get('predicted')} "
            f"P={d.get('retrieval_precision')} R={d.get('retrieval_recall')} "
            f"halluc={d.get('hallucinated')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
