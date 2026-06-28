"""Engine loader for the API layer.

Loads the trained model and the bundled demo signals once and exposes simple
accessors. Infrastructure/marshalling only; all signal-processing and ML logic
lives in :mod:`src` (the repo-root engine package). Analogous to the template's
``db.py`` -- a thin resource layer the blueprints depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.pipeline import Pipeline

from src import config
from src import model as model_module


@dataclass
class Engine:
    """The loaded model plus the bundled demo signals."""

    model: Pipeline
    fs: int
    sample_ids: list[str]
    conditions: list[str]
    signals: np.ndarray
    rpms: np.ndarray
    index: dict[str, int] = field(default_factory=dict)


_engine: Engine | None = None


def load_engine(
    model_path=config.MODEL_PATH, samples_path=config.SAMPLES_PATH
) -> Engine:
    """Load model + demo samples into the module-level singleton and return it.

    Raises:
        FileNotFoundError: if the model or samples bundle is missing.
    """
    global _engine
    clf = model_module.load(model_path)
    if not samples_path.exists():
        raise FileNotFoundError(
            f"No samples bundle at {samples_path}. Run `python scripts/train.py`."
        )
    data = np.load(samples_path, allow_pickle=False)
    ids = [str(x) for x in data["ids"]]
    _engine = Engine(
        model=clf,
        fs=int(data["fs"]),
        sample_ids=ids,
        conditions=[str(x) for x in data["conditions"]],
        signals=data["signals"].astype(np.float64),
        rpms=data["rpms"].astype(np.float64),
        index={sid: i for i, sid in enumerate(ids)},
    )
    return _engine


def get_engine() -> Engine | None:
    """Return the loaded engine, or ``None`` if it has not been loaded."""
    return _engine
