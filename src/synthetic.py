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


def inject_impulses(
    signal: np.ndarray,
    points: list[int],
    amplitude: float = 1.0,
    fs: int = config.DEFAULT_FS,
    res_hz: float = 3000.0,
    decay_ms: float = 3.0,
) -> np.ndarray:
    """Inject damped resonance bursts at the given sample indices.

    Each burst is a decaying sinusoid (a stylized bearing-impact transient)
    scaled to ``amplitude`` times the signal's std, so injecting defects at
    specific spots in an otherwise-healthy window raises its impulsiveness.
    """
    x = np.asarray(signal, dtype=np.float64).copy()
    n = x.shape[0]
    if not points or amplitude <= 0:
        return x
    sd = float(np.std(x)) or 1.0
    tau = decay_ms / 1000.0
    burst_len = max(8, int(fs * tau * 4))
    t = np.arange(burst_len) / fs
    burst = np.exp(-t / tau) * np.sin(2 * np.pi * res_hz * t)
    for p in points:
        p = int(p)
        if 0 <= p < n:
            end = min(n, p + burst_len)
            x[p:end] += amplitude * sd * burst[: end - p]
    return x


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


def random_augment(
    window: np.ndarray,
    rng: np.random.Generator | None = None,
    max_noise: float = 0.6,
) -> np.ndarray:
    """Add a random amount of Gaussian noise, for training-time augmentation.

    Noise-only (no amplitude scaling): scaling would shift RMS/peak, which are
    discriminative features here, and could mislabel a scaled-up healthy window
    as a fault. Noise teaches sensor-noise robustness without that risk.
    """
    rng = rng or np.random.default_rng()
    noise = float(rng.uniform(0.1, max_noise))
    return add_noise(np.asarray(window, dtype=np.float64), noise, rng=rng)
