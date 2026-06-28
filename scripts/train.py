"""Train the bearing-fault classifier end to end.

Pipeline: load recordings -> window -> extract features -> split by recording
(no window leakage) -> train -> report accuracy + confusion matrix -> refit on
all data and save the model, plus a small bundle of demo signals for the API.

Run from the repo root:

    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src import config, data_loader, model, synthetic  # noqa: E402
from src.features import FEATURE_NAMES, extract_features_batch  # noqa: E402

# Augmented (noisy/scaled) windows generated per class to improve robustness.
N_AUGMENT_PER_CLASS = 50
# Synthetic injected faults generated per fault class (periodic impulse trains),
# so the model catches the kind of fault the Fault Lab generates. A generous count
# across a wide severity range teaches it subtle faults too (the ones it misses).
N_SYNTH_FAULT_PER_CLASS = 120


def _augment_from(
    windows: np.ndarray,
    labels: np.ndarray,
    rpms: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    per_class: int = N_AUGMENT_PER_CLASS,
    noise: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build augmented feature rows from the given window indices.

    ``noise=None`` uses random scaling + noise (training augmentation); a fixed
    ``noise`` level applies that exact noise (for the robustness stress test).
    Sampling per class keeps it balanced. Returns ``(X_aug, y_aug)``.
    """
    idx = np.asarray(indices)
    sub_labels = labels[idx]
    aug_X: list[np.ndarray] = []
    aug_y: list[np.ndarray] = []
    for cond in config.CONDITIONS:
        pool = idx[sub_labels == cond]
        if pool.size == 0:
            continue
        chosen = rng.choice(pool, size=per_class, replace=True)
        if noise is None:
            wins = np.stack([synthetic.random_augment(windows[i], rng) for i in chosen])
        else:
            wins = np.stack([synthetic.add_noise(windows[i], noise, rng) for i in chosen])
        aug_X.append(extract_features_batch(wins, fs=config.DEFAULT_FS, rpms=rpms[chosen]))
        aug_y.append(np.full(per_class, cond))
    if not aug_X:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], dtype=str)
    return np.vstack(aug_X), np.concatenate(aug_y)


def _synthetic_faults_from(
    windows: np.ndarray,
    labels: np.ndarray,
    rpms: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    per_class: int = N_SYNTH_FAULT_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature rows by injecting periodic faults into HEALTHY windows.

    Teaches the classifier the same synthesized-fault signature the Fault Lab
    generates (a periodic impulse train at each fault's characteristic frequency),
    so injected faults are caught and classified instead of called normal. Only
    the given indices' healthy windows are used. Returns ``(X_aug, y_aug)``.
    """
    idx = np.asarray(indices)
    healthy_pool = idx[labels[idx] == "normal"]
    if healthy_pool.size == 0:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], dtype=str)
    aug_X: list[np.ndarray] = []
    aug_y: list[np.ndarray] = []
    for cond in config.CONDITIONS:
        if cond == "normal":
            continue
        chosen = rng.choice(healthy_pool, size=per_class, replace=True)
        wins = np.stack(
            [
                synthetic.fault_window(
                    windows[i], cond, rpm=float(rpms[i]), fs=config.DEFAULT_FS,
                    severity=float(rng.uniform(0.5, 2.5)), rng=rng,
                )
                for i in chosen
            ]
        )
        aug_X.append(extract_features_batch(wins, fs=config.DEFAULT_FS, rpms=rpms[chosen]))
        aug_y.append(np.full(per_class, cond))
    if not aug_X:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], dtype=str)
    return np.vstack(aug_X), np.concatenate(aug_y)


def _print_confusion_matrix(labels: list[str], matrix: list[list[int]]) -> None:
    width = max(len(s) for s in labels) + 2
    header = " " * width + "".join(f"{lab[:8]:>10}" for lab in labels)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(header)
    for lab, row in zip(labels, matrix):
        print(f"{lab:>{width}}" + "".join(f"{v:>10}" for v in row))


def _export_samples(
    groups: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    path: Path = config.SAMPLES_PATH,
) -> None:
    """Bundle one held-out raw signal per condition for the web demo."""
    test_groups = set(groups[test_idx])
    ids, conditions, rpms, signals = [], [], [], []

    for condition in config.CONDITIONS:
        # Prefer a recording from the held-out test set; fall back to any.
        in_cond = {g for g, lab in zip(groups, labels) if lab == condition}
        held_out = sorted(in_cond & test_groups) or sorted(in_cond)
        if not held_out:
            continue
        recording = held_out[0]
        mat_path = config.DATA_DIR / condition / f"{recording}.mat"
        signal, rpm = data_loader.load_mat(mat_path)

        # Centre slice of fixed length for a clean, representative waveform.
        n = config.SAMPLE_SIGNAL_LEN
        if signal.shape[0] > n:
            start = (signal.shape[0] - n) // 2
            signal = signal[start : start + n]

        ids.append(condition)
        conditions.append(condition)
        rpms.append(rpm)
        signals.append(signal.astype(np.float32))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array(ids),
        conditions=np.array(conditions),
        rpms=np.array(rpms, dtype=np.float64),
        signals=np.stack(signals),
        fs=np.array(config.DEFAULT_FS),
    )
    print(f"Saved {len(ids)} demo signals to {path}")


def main() -> int:
    print("Loading dataset from", config.DATA_DIR)
    windows, labels, groups, rpms = data_loader.load_dataset()
    print(f"  {windows.shape[0]} windows from {len(set(groups))} recordings")

    print("Extracting features ...")
    X = extract_features_batch(windows, fs=config.DEFAULT_FS, rpms=rpms)
    print(f"  feature matrix: {X.shape}")

    # Split by recording so windows from one recording never span train+test.
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, labels, groups))

    leak = set(groups[train_idx]) & set(groups[test_idx])
    assert not leak, f"window leakage detected across recordings: {leak}"
    print(
        f"  train recordings: {sorted(set(groups[train_idx]))}\n"
        f"  test  recordings: {sorted(set(groups[test_idx]))}"
    )

    # Data augmentation: add noisy/scaled windows (the same kind of variation the
    # random generator / real sensors produce) to the TRAINING split only.
    rng = np.random.default_rng(42)
    Xaug_tr, yaug_tr = _augment_from(windows, labels, rpms, train_idx, rng)
    Xsyn_tr, ysyn_tr = _synthetic_faults_from(windows, labels, rpms, train_idx, rng)
    X_tr = np.vstack([X[train_idx], Xaug_tr, Xsyn_tr])
    y_tr = np.concatenate([labels[train_idx], yaug_tr, ysyn_tr])
    print(
        f"\nAugmentation: {len(train_idx)} real + {len(yaug_tr)} noisy "
        f"+ {len(ysyn_tr)} synthetic injected faults = {len(y_tr)} training windows"
    )

    print("Training (held-out evaluation, augmented) ...")
    eval_model = model.train(X_tr, y_tr)
    metrics = model.evaluate(eval_model, X[test_idx], labels[test_idx])
    print(f"\nHeld-out (clean) accuracy: {metrics['accuracy']:.4f}\n")
    _print_confusion_matrix(metrics["labels"], metrics["confusion_matrix"])
    print("\nPer-class report:\n")
    print(metrics["report"])

    # Robustness sweep: accuracy on increasingly-noised held-out windows, with
    # vs without augmentation (honest - the noisy set comes from test recordings
    # the models never trained on). CWRU is very separable, so the gap only opens
    # at severe noise.
    baseline = model.train(X[train_idx], labels[train_idx])
    print("\nRobustness (accuracy on noised held-out windows):")
    print(f"  {'noise':>6} {'baseline':>10} {'augmented':>10}")
    for lvl in (0.6, 1.0, 1.5, 2.0):
        Xn, yn = _augment_from(windows, labels, rpms, test_idx, rng, noise=lvl)
        b = model.evaluate(baseline, Xn, yn)["accuracy"]
        a = model.evaluate(eval_model, Xn, yn)["accuracy"]
        print(f"  {lvl:>6.1f} {b:>10.4f} {a:>10.4f}")

    # Refit the final model on ALL data + augmentation, then save.
    print("\nRefitting final model on all data + augmentation ...")
    all_idx = np.arange(len(labels))
    Xaug_all, yaug_all = _augment_from(windows, labels, rpms, all_idx, rng)
    Xsyn_all, ysyn_all = _synthetic_faults_from(windows, labels, rpms, all_idx, rng)
    final_model = model.train(
        np.vstack([X, Xaug_all, Xsyn_all]),
        np.concatenate([labels, yaug_all, ysyn_all]),
    )
    saved = model.save(final_model)
    print(f"Saved model to {saved}")

    _export_samples(groups, labels, test_idx)
    print("\nDone. Start the API: cd infra/api && python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
