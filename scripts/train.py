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

from src import config, data_loader, model  # noqa: E402
from src.features import extract_features_batch  # noqa: E402


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

    print("\nTraining (held-out evaluation) ...")
    eval_model = model.train(X[train_idx], labels[train_idx])
    metrics = model.evaluate(eval_model, X[test_idx], labels[test_idx])
    print(f"\nHeld-out accuracy: {metrics['accuracy']:.4f}\n")
    _print_confusion_matrix(metrics["labels"], metrics["confusion_matrix"])
    print("\nPer-class report:\n")
    print(metrics["report"])

    # Refit the final model on ALL data for the best demo classifier, then save.
    print("Refitting final model on all data ...")
    final_model = model.train(X, labels)
    saved = model.save(final_model)
    print(f"Saved model to {saved}")

    _export_samples(groups, labels, test_idx)
    print("\nDone. Start the API with: uvicorn backend.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
