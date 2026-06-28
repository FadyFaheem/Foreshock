#!/usr/bin/env python3
"""Foreshock API server (Flask).

A thin HTTP layer over the :mod:`src` engine, following the template's
application-factory pattern. No database and no auth (stateless demo).

Run host-only (from this directory):

    cd infra/api && python app.py          # dev server (PORT defaults to 8000)
    cd infra/api && gunicorn app:app       # production WSGI

In the Podman pods this same file runs with ``PORT=5000`` (see infra/podman).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# infra/api -> repo root is two levels up; add it so ``import src`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify  # noqa: E402
from flask.json.provider import DefaultJSONProvider  # noqa: E402
from flask_cors import CORS  # noqa: E402

import db  # noqa: E402  (flat imports: this dir is on sys.path)
import engine  # noqa: E402
from ai import ai_bp  # noqa: E402
from health_routes import health_bp  # noqa: E402
from predict import api_bp  # noqa: E402
from stream import stream_bp  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderedJSONProvider(DefaultJSONProvider):
    """Preserve key insertion order in JSON responses."""

    sort_keys = False


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.json = OrderedJSONProvider(app)

    # No auth/cookies in v1; allow any origin (dev proxy / tunnel are same-origin).
    CORS(app)

    app.register_blueprint(api_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(stream_bp)

    # Load the trained model + demo samples once. If missing, the app still
    # boots and endpoints return 503 until `scripts/train.py` has been run.
    try:
        engine.load_engine()
        logger.info("Foreshock engine loaded.")
    except FileNotFoundError as exc:
        logger.warning("%s", exc)

    # Apply DB migrations if a database is reachable (no-op otherwise).
    try:
        if db.available():
            applied = db.run_migrations()
            logger.info("DB ready (migrations applied: %s).", applied or "none")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB migration check skipped: %s", exc)

    @app.get("/health")
    def health():
        return jsonify(status="healthy", service="foreshock-api")

    @app.get("/")
    def root():
        return jsonify(
            name="Foreshock API",
            endpoints=["/api/samples", "/api/signal/<id>", "/api/predict"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    # Default to 8000 for host dev (macOS reserves 5000 for AirPlay); the
    # Podman pods set PORT=5000 explicitly.
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_ENV", "development") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
