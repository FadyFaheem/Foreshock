"""Feature extraction for vibration windows.

Turns one signal window into a fixed-length, named feature vector combining:

- Time domain: RMS, peak, peak-to-peak, kurtosis, skewness, crest factor, std.
- Frequency domain (FFT): band energy around each bearing fault frequency and
  the spectral centroid.
- Envelope (Hilbert) analysis: peak energy at each bearing fault frequency.

The module also exposes spectrum helpers (:func:`fft_spectrum`,
:func:`envelope_spectrum`, :func:`downsample`) that the backend reuses so it
never has to implement analysis itself.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.stats

from src import config

_EPS = 1e-12

# Ordered feature names. extract_features returns values in exactly this order.
FEATURE_NAMES: tuple[str, ...] = (
    # --- time domain ---
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "crest_factor",
    "std",
    # --- frequency domain (FFT) ---
    "fft_band_BPFO",
    "fft_band_BPFI",
    "fft_band_BSF",
    "fft_band_FTF",
    "spectral_centroid",
    # --- envelope (Hilbert) spectrum peaks ---
    "env_peak_BPFO",
    "env_peak_BPFI",
    "env_peak_BSF",
    "env_peak_FTF",
)


# --------------------------------------------------------------------------
# Spectrum helpers (also used by the backend for charting)
# --------------------------------------------------------------------------
def fft_spectrum(
    window: np.ndarray, fs: int = config.DEFAULT_FS
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of a window.

    Returns ``(freqs_hz, magnitude)``. The mean is removed first so the DC bin
    does not dominate.
    """
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    n = x.shape[0]
    mags = np.abs(np.fft.rfft(x)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mags


def envelope_spectrum(
    window: np.ndarray, fs: int = config.DEFAULT_FS
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (Hilbert) spectrum of a window.

    The amplitude envelope is obtained from the analytic signal, de-meaned, and
    Fourier transformed. Bearing faults show up as peaks at the characteristic
    fault frequencies in this spectrum.
    """
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    envelope = np.abs(scipy.signal.hilbert(x - x.mean()))
    envelope = envelope - envelope.mean()
    n = envelope.shape[0]
    mags = np.abs(np.fft.rfft(envelope)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mags


def downsample(signal: np.ndarray, n_points: int) -> np.ndarray:
    """Uniformly subsample a 1-D signal to at most ``n_points`` for display."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if n_points <= 0 or x.shape[0] <= n_points:
        return x
    idx = np.linspace(0, x.shape[0] - 1, n_points).astype(int)
    return x[idx]


def _limit_and_downsample(
    freqs: np.ndarray,
    mags: np.ndarray,
    max_hz: float | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_hz is not None:
        keep = freqs <= max_hz
        freqs, mags = freqs[keep], mags[keep]
    if max_points and freqs.shape[0] > max_points:
        idx = np.linspace(0, freqs.shape[0] - 1, max_points).astype(int)
        freqs, mags = freqs[idx], mags[idx]
    return freqs, mags


def fft_spectrum_for_display(
    signal: np.ndarray,
    fs: int = config.DEFAULT_FS,
    max_hz: float | None = config.FFT_DISPLAY_MAX_HZ,
    max_points: int = config.SPECTRUM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT spectrum truncated to ``max_hz`` and downsampled for charting."""
    freqs, mags = fft_spectrum(signal, fs)
    return _limit_and_downsample(freqs, mags, max_hz, max_points)


def envelope_spectrum_for_display(
    signal: np.ndarray,
    fs: int = config.DEFAULT_FS,
    max_hz: float | None = config.ENVELOPE_DISPLAY_MAX_HZ,
    max_points: int = config.SPECTRUM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope spectrum truncated to ``max_hz`` and downsampled for charting."""
    freqs, mags = envelope_spectrum(signal, fs)
    return _limit_and_downsample(freqs, mags, max_hz, max_points)


# --------------------------------------------------------------------------
# Band helpers
# --------------------------------------------------------------------------
def _band_mask(
    freqs: np.ndarray, center: float, halfwidth: float
) -> np.ndarray:
    return (freqs >= center - halfwidth) & (freqs <= center + halfwidth)


def _band_energy(
    freqs: np.ndarray, mags: np.ndarray, center: float, halfwidth: float
) -> float:
    mask = _band_mask(freqs, center, halfwidth)
    if not mask.any():
        return 0.0
    return float(np.sum(mags[mask] ** 2))


def _band_peak(
    freqs: np.ndarray, mags: np.ndarray, center: float, halfwidth: float
) -> float:
    mask = _band_mask(freqs, center, halfwidth)
    if not mask.any():
        return 0.0
    return float(np.max(mags[mask]))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _time_features(x: np.ndarray) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    has_var = std > _EPS
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "kurtosis": float(scipy.stats.kurtosis(x)) if has_var else 0.0,
        "skewness": float(scipy.stats.skew(x)) if has_var else 0.0,
        "crest_factor": peak / rms if rms > _EPS else 0.0,
        "std": std,
    }


def _frequency_features(
    x: np.ndarray, fs: int, freqs_hz: dict[str, float], halfwidth: float
) -> dict[str, float]:
    freqs, mags = fft_spectrum(x, fs)
    total = float(np.sum(mags))
    centroid = float(np.sum(freqs * mags) / total) if total > _EPS else 0.0
    feats = {"spectral_centroid": centroid}
    for name in config.FAULT_FREQ_NAMES:
        feats[f"fft_band_{name}"] = _band_energy(
            freqs, mags, freqs_hz[name], halfwidth
        )
    return feats


def _envelope_features(
    x: np.ndarray, fs: int, freqs_hz: dict[str, float], halfwidth: float
) -> dict[str, float]:
    freqs, mags = envelope_spectrum(x, fs)
    return {
        f"env_peak_{name}": _band_peak(freqs, mags, freqs_hz[name], halfwidth)
        for name in config.FAULT_FREQ_NAMES
    }


def extract_features(
    window: np.ndarray,
    fs: int = config.DEFAULT_FS,
    rpm: float = config.DEFAULT_RPM,
) -> np.ndarray:
    """Extract the ordered feature vector for a single window.

    Args:
        window: 1-D signal window.
        fs: sampling rate in Hz.
        rpm: shaft speed used to locate the bearing fault frequencies.

    Returns:
        1-D float64 array of length ``len(FEATURE_NAMES)``.
    """
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    freqs_hz = config.fault_frequencies(rpm)
    halfwidth = config.BAND_HALFWIDTH_HZ

    values: dict[str, float] = {}
    values.update(_time_features(x))
    values.update(_frequency_features(x, fs, freqs_hz, halfwidth))
    values.update(_envelope_features(x, fs, freqs_hz, halfwidth))

    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_batch(
    windows: np.ndarray,
    fs: int = config.DEFAULT_FS,
    rpms: float | np.ndarray = config.DEFAULT_RPM,
) -> np.ndarray:
    """Extract features for a batch of windows.

    Args:
        windows: array of shape ``(n_windows, window_len)``.
        fs: sampling rate in Hz.
        rpms: a scalar shaft speed or a per-window array of shape ``(n_windows,)``.

    Returns:
        Array of shape ``(n_windows, len(FEATURE_NAMES))``.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 1:
        windows = windows[np.newaxis, :]
    n = windows.shape[0]

    rpm_arr = np.broadcast_to(np.asarray(rpms, dtype=np.float64), (n,))
    out = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i in range(n):
        out[i] = extract_features(windows[i], fs, float(rpm_arr[i]))
    return out
