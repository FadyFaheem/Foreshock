"""Train the v2 health model (autoencoder) and build the run-to-failure demo.

Trains a feature-space autoencoder on HEALTHY windows only, then:
- builds a reconstruction-error timeline over a run-to-failure sequence
  (real IMS snapshots if present under data/ims, else a CWRU-derived sequence),
- builds a 2-D embedding of every condition (healthy clustering, faults drifting),
and saves models/health_ae.joblib + models/health.npz for the API/UI.

    python scripts/train_health.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src import config, data_loader, health, ims_loader  # noqa: E402
from src.features import extract_features, extract_features_batch  # noqa: E402

HEALTH_MODEL_PATH = config.MODELS_DIR / "health_ae.joblib"
HEALTH_DATA_PATH = config.MODELS_DIR / "health.npz"

# How many points per phase in the demo timeline / per condition in the embedding.
_TIMELINE_PER_PHASE = 150
_EMBED_PER_CLASS = 150
_SMOOTH = 7


def _smooth(x: np.ndarray, k: int = _SMOOTH) -> np.ndarray:
    """Causal (trailing) moving average so a rise never smears earlier in time."""
    if x.shape[0] < k:
        return x
    out = np.array(x, dtype=np.float64)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[k - 1 :] = (csum[k:] - csum[:-k]) / k
    return out


def _alarm_index(errors: np.ndarray, threshold: float) -> int:
    over = np.where(errors > threshold)[0]
    return int(over[0]) if over.size else -1


def _build_ims(test_dir: Path):
    """Build a real run-to-failure timeline from IMS snapshots (chronological)."""
    print(f"Using NASA IMS run-to-failure data: {test_dir}")
    feats = []
    for name, signal in ims_loader.iter_run(test_dir):
        window = signal[: config.WINDOW_SIZE]
        feats.append(extract_features(window, fs=ims_loader.IMS_FS, rpm=2000.0))
    X = np.vstack(feats)
    n = X.shape[0]
    # Early life is assumed healthy; train on the first third.
    healthy = X[: max(5, n // 3)]
    model = health.fit_health_model(healthy)
    errors = model.errors(X)
    phase = np.array(
        ["healthy"] * (n // 3) + ["degrading"] * (n - n // 3), dtype=object
    )[:n]
    return model, X, errors, phase, "ims"


def _build_cwru():
    """Build a CWRU-derived run-to-failure timeline (healthy block -> fault block)."""
    print("Using CWRU-derived run-to-failure timeline (no IMS data present).")
    windows, labels, _groups, rpms = data_loader.load_dataset()
    X = extract_features_batch(windows, fs=config.DEFAULT_FS, rpms=rpms)

    healthy = X[labels == "normal"]
    model = health.fit_health_model(healthy)

    # Sequence: healthy windows, then an inner-race fault block (one "asset").
    fault_cond = "inner_race" if "inner_race" in set(labels) else sorted(set(labels) - {"normal"})[0]
    h = X[labels == "normal"][:_TIMELINE_PER_PHASE]
    f = X[labels == fault_cond][:_TIMELINE_PER_PHASE]
    timeline_X = np.vstack([h, f])
    errors = model.errors(timeline_X)
    phase = np.array(["healthy"] * len(h) + ["fault"] * len(f), dtype=object)
    return model, X, errors, phase, "cwru", labels


def _build_embedding(model: health.HealthModel, X: np.ndarray, labels: np.ndarray):
    xs, ys, conds = [], [], []
    rng = np.random.default_rng(42)
    for cond in config.CONDITIONS:
        idx = np.where(labels == cond)[0]
        if idx.size == 0:
            continue
        take = idx if idx.size <= _EMBED_PER_CLASS else rng.choice(idx, _EMBED_PER_CLASS, replace=False)
        pts = model.embed(X[take])
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())
        conds.extend([cond] * len(take))
    return np.array(xs), np.array(ys), np.array(conds, dtype=object)


def main() -> int:
    ims_dir = ims_loader.available()
    if ims_dir is not None:
        model, X, errors, phase, source = _build_ims(ims_dir)
        labels = phase  # IMS embedding uses phase as the label
        embed_x, embed_y, embed_cond = _build_embedding(model, X, phase)
    else:
        model, X, errors, phase, source, labels = _build_cwru()
        embed_x, embed_y, embed_cond = _build_embedding(model, X, labels)

    smooth = _smooth(errors)
    alarm = _alarm_index(smooth, model.threshold)

    health.save(model, HEALTH_MODEL_PATH)
    np.savez_compressed(
        HEALTH_DATA_PATH,
        timeline_error=errors.astype(np.float64),
        timeline_smooth=smooth.astype(np.float64),
        timeline_phase=phase,
        threshold=np.array(model.threshold),
        alarm_index=np.array(alarm),
        embed_x=embed_x.astype(np.float64),
        embed_y=embed_y.astype(np.float64),
        embed_condition=embed_cond,
        source=np.array(source),
    )
    print(f"Saved health model to {HEALTH_MODEL_PATH}")
    print(
        f"Saved health timeline + embedding to {HEALTH_DATA_PATH} "
        f"(source={source}, points={errors.shape[0]}, threshold={model.threshold:.4f}, "
        f"alarm_index={alarm})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
