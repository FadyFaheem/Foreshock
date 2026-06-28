"""Self-test: confirm the AI catches injected faults (run by bootstrap).

Builds a healthy window, then checks the two things the Fault Lab does:
  1. hand-injected impulses (Custom mode) are flagged as an anomaly, and
  2. a generated outer-race fault is classified as a fault.

Prints PASS/FAIL per check with the key numbers (kurtosis, classifier call) so a
miss is debuggable. Uses only the local detection path (classifier + features +
anomaly check - no LLM/DB), and always exits 0: it is a diagnostic, not a gate.

    python scripts/selftest_detect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "infra" / "api")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402


def _check(agent, label: str, signal: np.ndarray, fs: int, rpm: float) -> bool:
    """Run the detection decision on one window and print PASS/FAIL."""
    analysis = agent._analyze(np.asarray(signal, dtype=float), fs, rpm)
    cond, unclassified = agent._resolve_condition(analysis, None)
    caught = unclassified or cond != "normal"
    reported = "anomaly" if unclassified else cond
    print(
        f"  [{'PASS' if caught else 'FAIL'}] {label} -> {reported} "
        f"(kurtosis {analysis['features'].get('kurtosis', 0.0):.2f}, "
        f"classifier {analysis['condition']} {analysis['confidence']:.0%})"
    )
    return caught


def main() -> int:
    try:
        import agent
        from engine import get_engine, load_engine
        from src import synthetic
    except Exception as exc:  # noqa: BLE001
        print(f"selftest: import failed ({exc}); skipping")
        return 0

    if get_engine() is None:
        try:
            load_engine()
        except Exception as exc:  # noqa: BLE001
            print(f"selftest: model not loaded ({exc}); skipping")
            return 0
    eng = get_engine()
    if eng is None or "normal" not in eng.index:
        print("selftest: no healthy sample available; skipping")
        return 0

    try:
        i = eng.index["normal"]
        rpm, fs = float(eng.rpms[i]), eng.fs
        base = synthetic.random_window(eng.signals[i], rng=np.random.default_rng(0))
        custom = synthetic.inject_impulses(
            base, list(range(0, base.shape[0], 170)), amplitude=3.0
        )
        generated = synthetic.fault_window(base, "outer_race", rpm=rpm, fs=fs, severity=1.5)

        c1 = _check(agent, "injected impulses (Custom mode)", custom, fs, rpm)
        c2 = _check(agent, "generated outer-race fault", generated, fs, rpm)
    except Exception as exc:  # noqa: BLE001
        print(f"selftest: error during checks ({exc}); skipping")
        return 0

    if c1 and c2:
        print("selftest: PASS")
    else:
        print("selftest: FAIL (faults missed - retrain the models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
