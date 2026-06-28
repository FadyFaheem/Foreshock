"""Load CWRU bearing recordings, label them, and segment into windows.

Loading and windowing are kept as separate, individually testable functions.
The public surface:

- :func:`load_mat`        - read one .mat file -> (signal, rpm)
- :func:`load_mat_bytes`  - read a .mat from raw bytes (uploads)
- :func:`load_csv_bytes`  - read a single-column CSV from raw bytes (uploads)
- :func:`segment`         - split a 1-D signal into overlapping windows
- :func:`load_dataset`    - walk data/<condition>/*.mat -> windows/labels/groups
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import scipy.io

from src import config

# CWRU variable names are prefixed with the file number, e.g. ``X105_DE_time``
# and ``X105RPM``. Match by suffix so we do not depend on the number.
_DE_KEY = re.compile(r"DE_time$")
_RPM_KEY = re.compile(r"RPM$")


def _find_key(mat: dict, pattern: re.Pattern[str]) -> str | None:
    """Return the first non-private mat key whose name matches ``pattern``."""
    for key in mat:
        if isinstance(key, str) and not key.startswith("__") and pattern.search(key):
            return key
    return None


def _extract_signal_rpm(mat: dict) -> tuple[np.ndarray, float]:
    """Pull the drive-end signal (1-D) and RPM out of a loaded mat dict."""
    de_key = _find_key(mat, _DE_KEY)
    if de_key is None:
        raise ValueError("No drive-end ('*DE_time') signal found in .mat file")
    signal = np.asarray(mat[de_key], dtype=np.float64).reshape(-1)

    rpm_key = _find_key(mat, _RPM_KEY)
    if rpm_key is not None:
        rpm = float(np.asarray(mat[rpm_key], dtype=np.float64).reshape(-1)[0])
    else:
        rpm = config.DEFAULT_RPM
    return signal, rpm


def load_mat(path: str | Path) -> tuple[np.ndarray, float]:
    """Load a CWRU .mat file from disk.

    Returns:
        ``(signal, rpm)`` where ``signal`` is the 1-D drive-end accelerometer
        series (float64) and ``rpm`` is the shaft speed.
    """
    mat = scipy.io.loadmat(str(path))
    return _extract_signal_rpm(mat)


def load_mat_bytes(raw: bytes) -> tuple[np.ndarray, float]:
    """Load a CWRU .mat file from raw bytes (e.g. an HTTP upload)."""
    mat = scipy.io.loadmat(io.BytesIO(raw))
    return _extract_signal_rpm(mat)


def load_csv_bytes(raw: bytes) -> tuple[np.ndarray, float]:
    """Load a waveform from raw CSV bytes (first numeric column).

    Non-numeric lines (e.g. a header) are skipped. RPM is unknown for CSV
    uploads, so the configured default is returned.
    """
    text = raw.decode("utf-8", errors="ignore")
    values: list[float] = []
    for line in text.splitlines():
        token = line.strip().split(",")[0].strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue  # header or non-numeric row
    if not values:
        raise ValueError("No numeric samples found in CSV upload")
    return np.asarray(values, dtype=np.float64), config.DEFAULT_RPM


def segment(
    signal: np.ndarray,
    window: int = config.WINDOW_SIZE,
    overlap: float = config.OVERLAP,
) -> np.ndarray:
    """Split a 1-D signal into overlapping fixed-length windows.

    Args:
        signal: 1-D input series.
        window: window length in samples.
        overlap: fractional overlap in [0, 1); 0.5 means 50% overlap.

    Returns:
        Array of shape ``(n_windows, window)``. Empty ``(0, window)`` if the
        signal is shorter than one window.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if window <= 0:
        raise ValueError("window must be positive")

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = signal.shape[0]
    if n < window:
        return np.empty((0, window), dtype=np.float64)

    step = max(1, int(round(window * (1.0 - overlap))))
    starts = range(0, n - window + 1, step)
    return np.stack([signal[s : s + window] for s in starts])


def load_dataset(
    data_dir: str | Path = config.DATA_DIR,
    window: int = config.WINDOW_SIZE,
    overlap: float = config.OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load every recording under ``data_dir/<condition>/*.mat`` into windows.

    The condition is taken from the sub-directory name (see
    :data:`src.config.CONDITIONS`). A per-window ``group`` equal to the source
    recording id is returned so callers can split train/test by recording and
    avoid window leakage.

    Returns:
        ``(windows, labels, groups, rpms)`` with shapes ``(M, window)``,
        ``(M,)``, ``(M,)``, ``(M,)``.
    """
    data_dir = Path(data_dir)
    windows_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    groups_all: list[np.ndarray] = []
    rpms_all: list[np.ndarray] = []

    for condition in config.CONDITIONS:
        cond_dir = data_dir / condition
        if not cond_dir.is_dir():
            continue
        for mat_path in sorted(cond_dir.glob("*.mat")):
            signal, rpm = load_mat(mat_path)
            w = segment(signal, window, overlap)
            if w.shape[0] == 0:
                continue
            windows_all.append(w)
            labels_all.append(np.full(w.shape[0], condition, dtype=object))
            groups_all.append(np.full(w.shape[0], mat_path.stem, dtype=object))
            rpms_all.append(np.full(w.shape[0], rpm, dtype=np.float64))

    if not windows_all:
        raise FileNotFoundError(
            f"No .mat recordings found under {data_dir}. "
            "Run `python scripts/download_data.py` first."
        )

    windows = np.concatenate(windows_all, axis=0)
    labels = np.concatenate(labels_all).astype(str)
    groups = np.concatenate(groups_all).astype(str)
    rpms = np.concatenate(rpms_all)
    return windows, labels, groups, rpms
