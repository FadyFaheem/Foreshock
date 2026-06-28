"""Sanity tests for the feature-extraction engine."""

from __future__ import annotations

import numpy as np

from src import config
from src.features import (
    FEATURE_NAMES,
    envelope_spectrum,
    extract_features,
    extract_features_batch,
    fft_spectrum,
)


def _sine(freq_hz: float, fs: int = config.FS_12K, n: int = config.WINDOW_SIZE,
          amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def test_feature_vector_has_fixed_length():
    vec = extract_features(_sine(100.0))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec.dtype == np.float64


def test_rms_of_unit_sine_is_root_half():
    vec = extract_features(_sine(120.0, amplitude=1.0))
    rms = vec[FEATURE_NAMES.index("rms")]
    assert np.isclose(rms, 1 / np.sqrt(2), atol=1e-3)


def test_crest_factor_of_sine_is_root_two():
    vec = extract_features(_sine(150.0, amplitude=2.0))
    crest = vec[FEATURE_NAMES.index("crest_factor")]
    assert np.isclose(crest, np.sqrt(2), atol=1e-2)


def test_features_are_deterministic():
    window = _sine(90.0) + 0.1 * _sine(300.0)
    a = extract_features(window, rpm=1797)
    b = extract_features(window, rpm=1797)
    np.testing.assert_array_equal(a, b)


def test_all_zero_window_is_finite():
    vec = extract_features(np.zeros(config.WINDOW_SIZE))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_batch_shape_matches_inputs():
    windows = np.stack([_sine(f) for f in (80.0, 120.0, 160.0)])
    out = extract_features_batch(windows, rpms=1797)
    assert out.shape == (3, len(FEATURE_NAMES))


def test_batch_accepts_per_window_rpm():
    windows = np.stack([_sine(100.0), _sine(100.0)])
    out = extract_features_batch(windows, rpms=np.array([1797.0, 1730.0]))
    assert out.shape == (2, len(FEATURE_NAMES))


def test_fft_spectrum_peaks_at_input_frequency():
    freqs, mags = fft_spectrum(_sine(200.0))
    peak_freq = freqs[int(np.argmax(mags))]
    assert abs(peak_freq - 200.0) < 10.0


def test_envelope_spectrum_returns_matching_lengths():
    freqs, mags = envelope_spectrum(_sine(100.0))
    assert freqs.shape == mags.shape
    assert freqs.ndim == 1
