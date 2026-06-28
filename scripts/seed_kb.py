"""Seed the RAG knowledge base.

Applies DB migrations, then embeds a small bearing-fault knowledge corpus with
the local embedding model and stores it in Postgres/pgvector. Re-runnable: it
clears and re-inserts the corpus each time.

    python scripts/seed_kb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "infra" / "api"
for _p in (str(PROJECT_ROOT), str(API_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db  # noqa: E402
import llm  # noqa: E402

# Domain knowledge corpus. fault_type is used both for retrieval filtering and
# as the relevance label in the eval harness.
CORPUS: list[dict[str, str]] = [
    {
        "fault_type": "normal",
        "title": "Healthy bearing vibration signature",
        "source": "ISO 10816 / CWRU",
        "content": (
            "A healthy rolling-element bearing shows low broadband vibration with no "
            "dominant peaks at the characteristic fault frequencies (BPFO, BPFI, BSF, "
            "FTF). Time-domain kurtosis stays near 3 and the crest factor is low and "
            "stable. Rising RMS or kurtosis over time is an early sign of degradation."
        ),
    },
    {
        "fault_type": "inner_race",
        "title": "Inner race fault (BPFI)",
        "source": "Vibration diagnostics handbook",
        "content": (
            "An inner-race defect excites the Ball Pass Frequency Inner (BPFI ~ 5.4x "
            "shaft speed for an SKF 6205). Because the defect rotates with the shaft, "
            "the envelope spectrum shows BPFI with sidebands spaced at 1x shaft speed. "
            "Recommended actions: confirm via envelope analysis, trend severity, and "
            "schedule bearing replacement; inspect lubrication and alignment."
        ),
    },
    {
        "fault_type": "outer_race",
        "title": "Outer race fault (BPFO)",
        "source": "Vibration diagnostics handbook",
        "content": (
            "An outer-race defect produces strong peaks at the Ball Pass Frequency "
            "Outer (BPFO ~ 3.6x shaft speed for an SKF 6205), typically without "
            "sidebands since the defect is stationary relative to the load zone. "
            "Recommended actions: verify BPFO in the envelope spectrum, assess load "
            "zone, and plan replacement; check for contamination and overload."
        ),
    },
    {
        "fault_type": "ball",
        "title": "Ball/rolling-element fault (BSF)",
        "source": "Vibration diagnostics handbook",
        "content": (
            "A rolling-element defect excites the Ball Spin Frequency (BSF ~ 2.36x "
            "shaft speed) and often its 2x harmonic, modulated by the cage frequency "
            "(FTF). Energy can appear intermittently as the damaged ball enters and "
            "leaves the load zone. Recommended actions: confirm BSF/2xBSF in the "
            "envelope spectrum, trend, and replace the bearing; review lubrication."
        ),
    },
    {
        "fault_type": "general",
        "title": "Envelope (Hilbert) analysis for bearing faults",
        "source": "Signal processing notes",
        "content": (
            "Bearing impacts are weak and high-frequency; demodulating with the "
            "Hilbert transform (envelope analysis) reveals periodic energy at the "
            "characteristic fault frequencies that is hidden in the raw FFT. Peaks at "
            "BPFO/BPFI/BSF/FTF in the envelope spectrum are the primary evidence used "
            "to localize a defect to the outer race, inner race, ball, or cage."
        ),
    },
    {
        "fault_type": "general",
        "title": "Severity assessment and maintenance priority",
        "source": "ISO 10816 guidance",
        "content": (
            "Grade severity from the strength and trend of fault-frequency energy and "
            "overall RMS. Low/stable energy = monitor (low priority). Clear, growing "
            "peaks = plan maintenance (medium). High amplitude with rising trend and "
            "elevated kurtosis = act soon (high priority) to avoid secondary damage. "
            "Always corroborate the model's class with the envelope-spectrum evidence."
        ),
    },
    {
        "fault_type": "general",
        "title": "Drafting a maintenance work order",
        "source": "Reliability engineering practice",
        "content": (
            "A bearing work order should state the asset, the diagnosed condition, the "
            "supporting evidence (fault frequency and amplitude), a priority, and "
            "concrete actions: inspect and confirm, replace the bearing, verify "
            "lubrication and alignment, and re-baseline vibration after the repair."
        ),
    },
]


def main() -> int:
    print("Applying migrations ...")
    db.run_migrations()

    print("Clearing knowledge_base ...")
    db.execute("DELETE FROM knowledge_base")

    print(f"Embedding + inserting {len(CORPUS)} documents with {llm.EMBED_MODEL} ...")
    for doc in CORPUS:
        embedding = llm.embed(f"{doc['title']}. {doc['content']}")
        db.execute(
            "INSERT INTO knowledge_base (fault_type, title, content, source, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            (doc["fault_type"], doc["title"], doc["content"], doc["source"], embedding),
        )

    row = db.query("SELECT COUNT(*) AS n FROM knowledge_base", fetch_one=True)
    print(f"Done. knowledge_base now has {row['n']} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
