"""Loader for the NASA IMS bearing run-to-failure dataset (optional, for v2).

Each IMS test directory holds many timestamped ASCII snapshots (~1 s at ~20 kHz,
one whitespace-separated column per accelerometer channel). Health degrades over
the chronological sequence, so loading snapshots in order yields a real
run-to-failure timeline.

v2 demos on a CWRU-derived timeline when IMS is absent. To use real data, place
an IMS test set under ``data/ims/<test>/`` and run ``scripts/train_health.py``;
it picks up IMS automatically via :func:`available`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src import config

IMS_FS = 20_000  # Hz (IMS sampled at ~20 kHz)
IMS_DIR = config.DATA_DIR / "ims"


def available(ims_dir: str | Path = IMS_DIR) -> Path | None:
    """Return a test directory containing snapshots, or None if IMS is absent."""
    root = Path(ims_dir)
    if not root.is_dir():
        return None
    # Accept either data/ims/<test>/<snapshots> or data/ims/<snapshots>.
    candidates = [root, *[d for d in root.iterdir() if d.is_dir()]]
    for d in candidates:
        if any(p.is_file() and not p.name.startswith(".") for p in d.iterdir()):
            return d
    return None


def list_snapshots(test_dir: str | Path) -> list[Path]:
    """Chronologically sorted snapshot files (IMS names sort lexicographically)."""
    test_dir = Path(test_dir)
    files = [
        p for p in test_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name)


def load_snapshot(path: str | Path, channel: int = 0) -> np.ndarray:
    """Load one snapshot's accelerometer channel as a 1-D float array."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    return arr[:, channel].astype(np.float64)


def iter_run(test_dir: str | Path, channel: int = 0) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(snapshot_name, signal)`` in chronological order."""
    for path in list_snapshots(test_dir):
        yield path.name, load_snapshot(path, channel)
