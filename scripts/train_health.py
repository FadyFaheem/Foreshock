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

from src import config, data_loader, health, ims_loader, synthetic  # noqa: E402
from src.features import extract_features, extract_features_batch  # noqa: E402

HEALTH_MODEL_PATH = config.MODELS_DIR / "health_ae.joblib"
HEALTH_DATA_PATH = config.MODELS_DIR / "health.npz"

# How many points per phase in the demo timeline / per condition in the embedding.
_TIMELINE_PER_PHASE = 150
_EMBED_PER_CLASS = 150
_SMOOTH = 7

# (c) Harden the v2 health model with the random generator: we mix noise-augmented
# copies of the HEALTHY windows into the autoencoder's training set so the
# reconstruction-error indicator tolerates noisy-but-healthy signals (fewer false
# alarms) without the model ever seeing a fault during training.
_N_HEALTHY_AUGMENT = 200


def _augment_healthy(healthy_windows, healthy_rpms, rng, fs):
    """Feature rows for noise-augmented copies of the healthy windows (or None)."""
    healthy_windows = np.asarray(healthy_windows)
    if _N_HEALTHY_AUGMENT <= 0 or healthy_windows.shape[0] == 0:
        return None
    pick = rng.integers(0, healthy_windows.shape[0], size=_N_HEALTHY_AUGMENT)
    wins = np.stack([synthetic.random_augment(healthy_windows[i], rng) for i in pick])
    rpms = np.asarray(healthy_rpms, dtype=float)[pick]
    return extract_features_batch(wins, fs=fs, rpms=rpms)


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
    feats, windows = [], []
    for _name, signal in ims_loader.iter_run(test_dir):
        window = signal[: config.WINDOW_SIZE]
        windows.append(window)
        feats.append(extract_features(window, fs=ims_loader.IMS_FS, rpm=2000.0))
    X = np.vstack(feats)
    W = np.stack(windows)
    n = X.shape[0]
    # Early life is assumed healthy; train the autoencoder on the first third only.
    healthy_count = max(5, n // 3)
    healthy = X[:healthy_count]
    rng = np.random.default_rng(0)
    aug = _augment_healthy(W[:healthy_count], np.full(healthy_count, 2000.0), rng, ims_loader.IMS_FS)
    if aug is not None:
        healthy = np.vstack([healthy, aug])
    model = health.fit_health_model(healthy)
    errors = model.errors(X)
    phase = np.array(
        ["healthy"] * healthy_count + ["degrading"] * (n - healthy_count), dtype=object
    )
    return model, X, errors, phase, "ims"


def _cwru_scorer() -> tuple[health.HealthModel, np.ndarray, np.ndarray]:
    """Train the scoring autoencoder on CWRU healthy windows (+ noise augmentation).

    This is the model that gets SAVED. It scores the bundled CWRU samples and the
    Fault-Lab windows at request time (the anomaly net + /api/health/sample), so its
    reconstruction error must be calibrated to *that* data - not IMS, whose scale is
    different. Returns ``(model, X, labels)`` for reuse by the embedding / timeline.
    """
    windows, labels, _groups, rpms = data_loader.load_dataset()
    windows = np.asarray(windows)
    labels = np.asarray(labels)
    rpms = np.asarray(rpms)
    X = extract_features_batch(windows, fs=config.DEFAULT_FS, rpms=rpms)

    healthy_mask = labels == "normal"
    healthy = X[healthy_mask]
    rng = np.random.default_rng(0)
    aug = _augment_healthy(windows[healthy_mask], rpms[healthy_mask], rng, config.DEFAULT_FS)
    if aug is not None:
        healthy = np.vstack([healthy, aug])
    return health.fit_health_model(healthy), X, labels


def _cwru_timeline(
    model: health.HealthModel, X: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A CWRU-derived run-to-failure timeline (healthy block -> fault block)."""
    fault_cond = (
        "inner_race" if "inner_race" in set(labels)
        else sorted(set(labels) - {"normal"})[0]
    )
    h = X[labels == "normal"][:_TIMELINE_PER_PHASE]
    f = X[labels == fault_cond][:_TIMELINE_PER_PHASE]
    errors = model.errors(np.vstack([h, f]))
    phase = np.array(["healthy"] * len(h) + ["fault"] * len(f), dtype=object)
    return errors, phase


def _build_embedding(model: health.HealthModel, X: np.ndarray, labels: np.ndarray):
    xs, ys, conds = [], [], []
    rng = np.random.default_rng(42)
    # Iterate the label set actually present (CWRU conditions, or IMS phases).
    for cond in dict.fromkeys(np.asarray(labels).tolist()):
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
    # The SAVED scoring autoencoder is always CWRU-calibrated: it scores the bundled
    # CWRU samples and Fault-Lab windows at request time (the anomaly net and
    # /api/health/sample), so its reconstruction error must be meaningful for *that*
    # data, not for IMS.
    print("Training the CWRU-calibrated scoring autoencoder ...")
    scoring_model, X, labels = _cwru_scorer()
    # Save it FIRST. The anomaly net depends on this file, so it must persist even
    # if the optional IMS timeline build below fails (e.g. a partial IMS download).
    health.save(scoring_model, HEALTH_MODEL_PATH)
    print(f"Saved CWRU-calibrated scoring model to {HEALTH_MODEL_PATH}")

    # Run-to-failure timeline + embedding (display only). Prefer real IMS data; fall
    # back to a CWRU-derived timeline on ANY IMS problem so it never blocks the save.
    timeline = None
    ims_dir = ims_loader.available()
    if ims_dir is not None:
        try:
            tmodel, ims_X, errors, phase, source = _build_ims(ims_dir)
            ex, ey, ec = _build_embedding(tmodel, ims_X, phase)
            timeline = (errors, phase, source, ex, ey, ec, float(tmodel.threshold))
        except Exception as exc:  # noqa: BLE001
            print(f"IMS timeline build failed ({exc}); using a CWRU timeline.")
    if timeline is None:
        print("Using a CWRU-derived timeline + embedding.")
        errors, phase = _cwru_timeline(scoring_model, X, labels)
        ex, ey, ec = _build_embedding(scoring_model, X, labels)
        timeline = (errors, phase, "cwru", ex, ey, ec, float(scoring_model.threshold))

    errors, phase, source, ex, ey, ec, threshold = timeline
    smooth = _smooth(errors)
    alarm = _alarm_index(smooth, threshold)
    np.savez_compressed(
        HEALTH_DATA_PATH,
        timeline_error=errors.astype(np.float64),
        timeline_smooth=smooth.astype(np.float64),
        timeline_phase=phase,
        threshold=np.array(threshold),
        alarm_index=np.array(alarm),
        embed_x=ex.astype(np.float64),
        embed_y=ey.astype(np.float64),
        embed_condition=ec,
        source=np.array(source),
    )
    print(
        f"Saved health timeline + embedding to {HEALTH_DATA_PATH} "
        f"(timeline source={source}, points={errors.shape[0]}, "
        f"threshold={threshold:.4f}, alarm_index={alarm})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
