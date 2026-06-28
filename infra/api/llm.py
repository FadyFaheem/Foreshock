"""Local LLM + embedding client (Ollama) with AI observability.

Thin wrapper over the Ollama HTTP API. Every call is timed and its latency +
token usage (+ optional retrieval quality) are written to the ``llm_calls``
table for the observability panel. Everything degrades gracefully: if Ollama is
unreachable, :class:`LLMUnavailable` is raised and callers return a clear error.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

import db

logger = logging.getLogger(__name__)


def _host() -> str:
    h = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not h.startswith("http"):
        h = f"http://{h}"
    return h.rstrip("/")


LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))


class LLMUnavailable(RuntimeError):
    """Raised when the Ollama server or a model is not reachable."""


def _log_call(
    operation: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    retrieval_score: float | None = None,
    ok: bool = True,
) -> None:
    """Best-effort observability record; never raises."""
    try:
        total = (prompt_tokens or 0) + (completion_tokens or 0)
        db.execute(
            "INSERT INTO llm_calls (operation, model, latency_ms, prompt_tokens, "
            "completion_tokens, total_tokens, retrieval_score, ok) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                operation,
                model,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total or None,
                retrieval_score,
                ok,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - observability must not break calls
        logger.debug("Could not log llm_call: %s", exc)


def available() -> bool:
    """True if the Ollama server responds."""
    try:
        requests.get(f"{_host()}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama unavailable at %s: %s", _host(), exc)
        return False


def embed(text: str) -> list[float]:
    """Return the embedding vector for ``text`` (logs latency)."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{_host()}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
    except Exception as exc:  # noqa: BLE001
        _log_call("embed", EMBED_MODEL, (time.perf_counter() - t0) * 1000, ok=False)
        raise LLMUnavailable(f"embedding failed: {exc}") from exc
    _log_call("embed", EMBED_MODEL, (time.perf_counter() - t0) * 1000)
    return vec


def chat(
    system: str,
    user: str,
    as_json: bool = True,
    retrieval_score: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run a single-turn chat completion. Returns (content, usage).

    ``usage`` has prompt/completion/total token counts and latency_ms.
    """
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    if as_json:
        payload["format"] = "json"

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{_host()}/api/chat", json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        _log_call(
            "chat", LLM_MODEL, (time.perf_counter() - t0) * 1000,
            retrieval_score=retrieval_score, ok=False,
        )
        raise LLMUnavailable(f"chat failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    content = data.get("message", {}).get("content", "")
    usage = {
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "latency_ms": latency_ms,
        "model": LLM_MODEL,
    }
    _log_call(
        "chat", LLM_MODEL, latency_ms,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        retrieval_score=retrieval_score,
    )
    return content, usage


def chat_json(
    system: str, user: str, retrieval_score: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like :func:`chat` but parses the model's JSON output. Returns (obj, usage)."""
    content, usage = chat(system, user, as_json=True, retrieval_score=retrieval_score)
    try:
        return json.loads(content), usage
    except json.JSONDecodeError:
        # Best-effort salvage of the first {...} block.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1]), usage
            except json.JSONDecodeError:
                pass
        return {"_raw": content}, usage
