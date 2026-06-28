"""Unsupervised health indicator: a feature-space autoencoder.

Trained on HEALTHY windows only. Reconstruction error rises as a bearing
degrades, giving an early-warning health indicator (the v2 "catch failures
earlier" story), and a 2-D PCA embedding shows healthy data clustering with
faults drifting away.

Implemented with scikit-learn - a small MLP autoencoder (a real neural net with
a 3-unit bottleneck) plus PCA - so there is no heavy deep-learning dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class HealthModel:
    """A trained autoencoder + PCA embedding + alarm threshold."""

    scaler: StandardScaler
    autoencoder: MLPRegressor
    pca: PCA
    threshold: float

    def errors(self, X: np.ndarray) -> np.ndarray:
        """Per-row reconstruction error (mean squared error in scaled space)."""
        xs = self.scaler.transform(np.atleast_2d(X))
        recon = self.autoencoder.predict(xs)
        return np.mean((xs - recon) ** 2, axis=1)

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Project feature vectors to 2-D for the embedding plot."""
        return self.pca.transform(self.scaler.transform(np.atleast_2d(X)))

    def health_score(self, X: np.ndarray) -> np.ndarray:
        """0..1 health-degradation score (error relative to the alarm threshold)."""
        return self.errors(X) / (self.threshold or 1.0)


def fit_health_model(
    X_healthy: np.ndarray,
    bottleneck: int = 3,
    threshold_percentile: float = 99.0,
    random_state: int = 42,
) -> HealthModel:
    """Train the autoencoder on healthy feature vectors only.

    Args:
        X_healthy: ``(n_healthy, n_features)`` feature matrix from healthy windows.
        bottleneck: size of the compressing hidden layer.
        threshold_percentile: healthy-error percentile used as the alarm threshold.
    """
    scaler = StandardScaler().fit(X_healthy)
    xs = scaler.transform(X_healthy)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(8, bottleneck, 8),
        activation="tanh",
        solver="adam",
        max_iter=3000,
        random_state=random_state,
    )
    autoencoder.fit(xs, xs)

    pca = PCA(n_components=2, random_state=random_state).fit(xs)

    recon = autoencoder.predict(xs)
    healthy_err = np.mean((xs - recon) ** 2, axis=1)
    threshold = float(np.percentile(healthy_err, threshold_percentile))

    return HealthModel(scaler=scaler, autoencoder=autoencoder, pca=pca, threshold=threshold)


def save(model: HealthModel, path) -> None:
    joblib.dump(model, path)


def load(path) -> HealthModel:
    return joblib.load(path)
