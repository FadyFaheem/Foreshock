"""Pytest fixtures for the Flask API.

These tests exercise the real engine (the trained model + bundled samples). If
the model has not been trained yet, the API-level tests skip themselves rather
than fail. DB-free: there is no database in this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1]  # infra/api
PROJECT_ROOT = API_DIR.parents[1]              # repo root
for _p in (str(API_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app import create_app  # noqa: E402


@pytest.fixture()
def app():
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
