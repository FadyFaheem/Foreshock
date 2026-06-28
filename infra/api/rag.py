"""Retrieval over the pgvector knowledge base.

Embeds a query with the local embedding model and returns the nearest knowledge
chunks by cosine similarity. Pure retrieval; generation lives in :mod:`llm`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import db
import llm


def kb_count() -> int:
    try:
        row = db.query("SELECT COUNT(*) AS n FROM knowledge_base", fetch_one=True)
        return int(row["n"]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def retrieve(query_text: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Return the top-k knowledge chunks for ``query_text`` (cosine similarity).

    Each item: id, fault_type, title, content, source, score (1 = identical).
    Requires the embedding model (raises :class:`llm.LLMUnavailable`) and the DB.
    """
    # pgvector's psycopg2 adapter expects a numpy array (a plain list adapts to
    # numeric[], which has no <=> operator against vector).
    embedding = np.asarray(llm.embed(query_text), dtype=np.float32)
    rows = db.query(
        "SELECT id, fault_type, title, content, source, "
        "1 - (embedding <=> %s) AS score "
        "FROM knowledge_base "
        "ORDER BY embedding <=> %s "
        "LIMIT %s",
        (embedding, embedding, top_k),
    )
    return [
        {
            "id": r["id"],
            "fault_type": r["fault_type"],
            "title": r["title"],
            "content": r["content"],
            "source": r["source"],
            "score": float(r["score"]),
        }
        for r in rows
    ]


def format_context(docs: list[dict[str, Any]]) -> str:
    """Render retrieved docs as a numbered context block for a prompt."""
    return "\n\n".join(
        f"[{i + 1}] ({d['fault_type']}) {d['title']}\n{d['content']}"
        for i, d in enumerate(docs)
    )
