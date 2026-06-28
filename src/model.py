"""Train, evaluate, persist, and load the bearing-fault classifier.

The estimator is a scikit-learn :class:`~sklearn.pipeline.Pipeline` of a
:class:`StandardScaler` followed by a :class:`RandomForestClassifier`, so the
saved artifact is fully self-contained (scaling parameters travel with it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def build_pipeline(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    """Construct the scaler + RandomForest pipeline.

    ``class_weight='balanced'`` compensates for the CWRU normal-baseline
    recordings being longer (and so producing more windows) than the fault
    recordings.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Pipeline:
    """Fit a new pipeline on ``(X, y)`` and return it."""
    model = build_pipeline(**kwargs)
    model.fit(X, y)
    return model


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Evaluate ``model`` on ``(X, y)``.

    Returns a dict with accuracy, the ordered class labels, the confusion
    matrix (rows = true, cols = predicted), and a text per-class report.
    """
    y_pred = model.predict(X)
    labels = [c for c in config.CONDITIONS if c in set(y)]
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "report": classification_report(y, y_pred, labels=labels, zero_division=0),
    }


def save(model: Pipeline, path: str | Path = config.MODEL_PATH) -> Path:
    """Persist ``model`` to ``path`` with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: str | Path = config.MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one with `python scripts/train.py`."
        )
    return joblib.load(path)
