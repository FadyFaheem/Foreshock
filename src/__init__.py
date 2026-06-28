"""Foreshock core engine: signal processing and ML for bearing-fault detection.

This package is intentionally UI-agnostic. It contains all signal-processing and
machine-learning logic and has no dependency on any web framework. The backend
imports from here; it never implements analysis itself.
"""

__all__ = ["config", "data_loader", "features", "model"]
