"""Configuration constants for the Foreshock engine.

This module holds *only* constants and a tiny helper that turns shaft speed into
the characteristic bearing fault frequencies. It contains no analysis logic and
no I/O. Everything here is shared by the data loader, feature extractor, model,
the training script, and (read-only) the backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# --- Project paths --------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
SAMPLES_PATH: Path = MODELS_DIR / "samples.npz"

# --- Sampling -------------------------------------------------------------
FS_12K: int = 12_000  # Hz; 12 kHz drive-end accelerometer data (the v1 dataset)
FS_48K: int = 48_000  # Hz; 48 kHz drive-end data (documented, unused in v1)
DEFAULT_FS: int = FS_12K

# Fallback shaft speed (RPM) for recordings/uploads that carry no RPM variable.
# 1797 RPM is the CWRU 0 HP baseline speed.
DEFAULT_RPM: float = 1797.0

# --- Windowing ------------------------------------------------------------
WINDOW_SIZE: int = 2048  # samples per analysis window (~0.17 s at 12 kHz)
OVERLAP: float = 0.5     # fractional overlap between consecutive windows

# --- Display / bundled demo samples (used by the web API) -----------------
SAMPLE_SIGNAL_LEN: int = 12_000        # raw samples bundled per demo signal (~1 s)
WAVEFORM_POINTS: int = 3_000           # downsampled points for the waveform chart
SPECTRUM_POINTS: int = 2_000           # downsampled points for spectrum charts
FFT_DISPLAY_MAX_HZ: float | None = None   # None -> show up to the Nyquist limit
ENVELOPE_DISPLAY_MAX_HZ: float = 500.0    # envelope spectrum is low-frequency

# --- Conditions / labels --------------------------------------------------
# Canonical ordering used everywhere (dataset dirs, label encoding, UI).
CONDITIONS: tuple[str, ...] = ("normal", "inner_race", "outer_race", "ball")
CONDITION_LABELS: dict[str, str] = {
    "normal": "Normal (healthy)",
    "inner_race": "Inner race fault",
    "outer_race": "Outer race fault",
    "ball": "Ball fault",
}

# --- Bearing geometry: SKF 6205-2RS JEM (CWRU drive-end bearing) -----------
@dataclass(frozen=True)
class BearingGeometry:
    """Geometry of a rolling-element bearing (inches, degrees)."""

    n_balls: int = 9
    ball_diameter_in: float = 0.3126
    pitch_diameter_in: float = 1.537
    contact_angle_deg: float = 0.0


DRIVE_END_BEARING = BearingGeometry()

# Characteristic fault-frequency names, in canonical order.
FAULT_FREQ_NAMES: tuple[str, ...] = ("BPFO", "BPFI", "BSF", "FTF")

# Half-width (Hz) of the band searched around each fault frequency when
# computing band energy (FFT) and peak energy (envelope spectrum).
BAND_HALFWIDTH_HZ: float = 5.0


def _fault_multipliers(g: BearingGeometry) -> dict[str, float]:
    """Fault-frequency multipliers (x shaft rate) from bearing geometry.

    BPFO = (n/2)(1 - d/D cos a)        outer race
    BPFI = (n/2)(1 + d/D cos a)        inner race
    BSF  = (D/2d)(1 - (d/D cos a)^2)   ball spin
    FTF  = (1/2)(1 - d/D cos a)        cage / fundamental train
    """
    ratio = (g.ball_diameter_in / g.pitch_diameter_in) * math.cos(
        math.radians(g.contact_angle_deg)
    )
    n = g.n_balls
    return {
        "BPFO": (n / 2.0) * (1.0 - ratio),
        "BPFI": (n / 2.0) * (1.0 + ratio),
        "BSF": (g.pitch_diameter_in / (2.0 * g.ball_diameter_in)) * (1.0 - ratio**2),
        "FTF": 0.5 * (1.0 - ratio),
    }


FAULT_MULTIPLIERS: dict[str, float] = _fault_multipliers(DRIVE_END_BEARING)


def fault_frequencies(rpm: float) -> dict[str, float]:
    """Return the characteristic bearing fault frequencies (Hz) for a shaft speed.

    Args:
        rpm: shaft rotational speed in revolutions per minute.

    Returns:
        Mapping of BPFO/BPFI/BSF/FTF to frequency in Hz.
    """
    shaft_hz = rpm / 60.0
    return {name: mult * shaft_hz for name, mult in FAULT_MULTIPLIERS.items()}
