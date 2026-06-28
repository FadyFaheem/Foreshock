"""Randomized test-window generation for the interactive accuracy tester.

Takes a labeled signal and produces a random fixed-length window, optionally
adding Gaussian noise so users can stress-test the classifier's robustness.
Pure signal logic; the backend only marshals.
"""

from __future__ import annotations

import numpy as np

from src import config


def random_window(
    signal: np.ndarray,
    window: int = config.WINDOW_SIZE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a random fixed-length slice of ``signal``.

    If the signal is shorter than ``window`` it is zero-padded to length.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.shape[0] < window:
        return np.pad(x, (0, window - x.shape[0]))
    start = int(rng.integers(0, x.shape[0] - window + 1))
    return x[start : start + window]


def add_noise(
    window: np.ndarray,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise scaled to a fraction of the window's std.

    ``noise_level`` is clamped to [0, 1]; 0 returns the window unchanged.
    """
    noise_level = float(max(0.0, min(1.0, noise_level)))
    if noise_level <= 0.0:
        return window
    rng = rng or np.random.default_rng()
    std = float(np.std(window)) or 1.0
    return window + rng.normal(0.0, noise_level * std, size=window.shape)
