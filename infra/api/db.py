"""PostgreSQL + pgvector access layer.

A thin, pooled DB helper modeled on the template's ``db.py``. Everything is
degrade-friendly: if the database is unreachable, :func:`available` returns
False and callers return a 503 instead of crashing. Vector columns use the
``pgvector`` psycopg2 adapter so Python lists round-trip as ``vector``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

# Default migrations dir: /migrations in containers, else the repo's infra/database.
_DEFAULT_MIGRATIONS = str(Path(__file__).resolve().parents[1] / "database")

_pool: ThreadedConnectionPool | None = None


def _conn_kwargs() -> dict[str, Any]:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "foreshock"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "3")),
    }


def get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        # The pool is per worker *process*; size it >= gunicorn --threads so
        # concurrent request threads don't hit "connection pool exhausted".
        # Default 16 matches the prod --threads; override with DB_POOL_MAX.
        maxconn = int(os.getenv("DB_POOL_MAX", "16"))
        _pool = ThreadedConnectionPool(minconn=1, maxconn=maxconn, **_conn_kwargs())
    return _pool


@contextmanager
def get_conn():
    """Yield a pooled connection with the pgvector type registered."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def available() -> bool:
    """True if the database is reachable."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database unavailable: %s", exc)
        return False


def query(
    sql: str, params: tuple | None = None, fetch_one: bool = False
) -> Any:
    """Run a SELECT and return list[dict] (or a single dict with fetch_one)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch_one else cur.fetchall()


def execute(sql: str, params: tuple | None = None, returning: bool = False) -> Any:
    """Run an INSERT/UPDATE/DELETE; optionally return the RETURNING row."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            if returning:
                return cur.fetchone()
            return None


def run_migrations(migrations_dir: str | None = None) -> list[str]:
    """Apply numbered *.sql migrations (skipping 000-*) tracked in schema_migrations.

    Returns the list of versions applied this run. Safe to call repeatedly.
    """
    directory = Path(migrations_dir or os.getenv("MIGRATIONS_DIR", _DEFAULT_MIGRATIONS))
    if not directory.is_dir():
        logger.warning("Migrations dir not found: %s", directory)
        return []

    applied: list[str] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version VARCHAR(255) PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
            )
            cur.execute("SELECT version FROM schema_migrations")
            done = {r[0] for r in cur.fetchall()}

        for path in sorted(directory.glob("*.sql")):
            version = path.stem
            if version.startswith("000-") or version in done:
                continue
            logger.info("Applying migration %s", version)
            with conn.cursor() as cur:
                cur.execute(path.read_text())
            applied.append(version)
    return applied


# Convenience re-export for callers building jsonb params.
__all__ = ["Json", "available", "execute", "get_conn", "query", "run_migrations"]
