"""v3 live sensor feed over Kafka.

A producer streams signal windows to a Kafka topic (simulating a live sensor);
a background consumer thread in the API runs inference on each window and pushes
results to connected browsers over Server-Sent Events (SSE).

Degrades gracefully: if kafka-python or the broker is unavailable, the status
endpoint reports it and /simulate returns 503; the SSE endpoint still connects.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque

import numpy as np
from flask import Blueprint, Response, jsonify, request

from engine import get_engine
from src import config, synthetic
from src.features import extract_features_batch

try:  # optional dependency - the app must boot even if it's missing
    from kafka import KafkaConsumer, KafkaProducer, TopicPartition
except Exception:  # noqa: BLE001
    KafkaConsumer = KafkaProducer = TopicPartition = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

stream_bp = Blueprint("stream", __name__, url_prefix="/api/stream")

TOPIC = os.getenv("KAFKA_TOPIC", "vibration")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

_subscribers: list[queue.Queue] = []
_subs_lock = threading.Lock()
_recent: deque = deque(maxlen=50)
_consumer_started = False
_producer_stop = threading.Event()
_producer_thread: threading.Thread | None = None


def _label(condition: str) -> str:
    return config.CONDITION_LABELS.get(condition, condition)


def kafka_available() -> bool:
    if KafkaProducer is None:
        return False
    try:
        p = KafkaProducer(bootstrap_servers=BOOTSTRAP, request_timeout_ms=3000)
        p.close(timeout=2)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kafka unavailable at %s: %s", BOOTSTRAP, exc)
        return False


def _publish(result: dict) -> None:
    _recent.append(result)
    with _subs_lock:
        for q in list(_subscribers):
            try:
                q.put_nowait(result)
            except queue.Full:
                pass


def _infer(value: dict) -> dict | None:
    eng = get_engine()
    if eng is None:
        return None
    window = np.asarray(value["window"], dtype=np.float64)
    rpm = float(value.get("rpm", config.DEFAULT_RPM))
    fs = int(value.get("fs", eng.fs))
    X = extract_features_batch(window[np.newaxis, :], fs=fs, rpms=rpm)
    proba = eng.model.predict_proba(X)[0]
    classes = [str(c) for c in eng.model.classes_]
    best = int(np.argmax(proba))
    pred = classes[best]
    actual = value.get("condition")
    return {
        "ts": value.get("ts", time.time()),
        "actual": actual,
        "actual_label": _label(actual) if actual else None,
        "prediction": pred,
        "prediction_label": _label(pred),
        "confidence": float(proba[best]),
        "rms": float(np.sqrt(np.mean(window**2))),
        "correct": (actual == pred) if actual else None,
    }


def _consume_loop() -> None:
    """Background thread: consume windows, run inference, fan out to SSE."""
    while True:
        if KafkaConsumer is None:
            return
        try:
            # Manual partition assignment (no consumer group) avoids rebalance
            # stalls and gives clean "live tail" semantics for a single consumer.
            consumer = KafkaConsumer(
                bootstrap_servers=BOOTSTRAP,
                value_deserializer=lambda b: json.loads(b.decode()),
                enable_auto_commit=False,
            )
            tp = TopicPartition(TOPIC, 0)
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            logger.info("Kafka consumer assigned %s (live tail).", TOPIC)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kafka consumer connect failed: %s", exc)
            time.sleep(3)
            continue
        try:
            while True:
                records = consumer.poll(timeout_ms=1000)
                for _tp, msgs in records.items():
                    for m in msgs:
                        result = _infer(m.value)
                        if result:
                            _publish(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kafka consumer error, reconnecting: %s", exc)
            time.sleep(3)
        finally:
            try:
                consumer.close()
            except Exception:  # noqa: BLE001
                pass


def start_consumer() -> None:
    """Start the background consumer once (no-op if kafka-python is missing)."""
    global _consumer_started
    if _consumer_started or KafkaConsumer is None:
        return
    _consumer_started = True
    threading.Thread(target=_consume_loop, name="kafka-consumer", daemon=True).start()
    logger.info("Kafka consumer thread started (topic=%s).", TOPIC)


def _produce(count: int, interval: float) -> None:
    eng = get_engine()
    if eng is None or KafkaProducer is None:
        return
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Producer connect failed: %s", exc)
        return
    rng = np.random.default_rng()
    try:
        for _ in range(count):
            if _producer_stop.is_set():
                break
            idx = int(rng.integers(len(eng.conditions)))
            window = synthetic.random_window(eng.signals[idx], rng=rng)
            producer.send(
                TOPIC,
                {
                    "window": window.tolist(),
                    "rpm": float(eng.rpms[idx]),
                    "fs": eng.fs,
                    "condition": eng.conditions[idx],
                    "ts": time.time(),
                },
            )
            time.sleep(interval)
        producer.flush()
    finally:
        producer.close()


@stream_bp.get("/status")
def status():
    return jsonify(
        kafka=kafka_available(),
        topic=TOPIC,
        recent=len(_recent),
        streaming=bool(_producer_thread and _producer_thread.is_alive()),
    )


@stream_bp.post("/simulate")
def simulate():
    """Start a simulated live feed: stream N random windows to Kafka."""
    global _producer_thread
    if get_engine() is None:
        return jsonify(error="Model not loaded. Run scripts/train.py."), 503
    if not kafka_available():
        return jsonify(error="Kafka unavailable. Is the broker running?"), 503
    if _producer_thread and _producer_thread.is_alive():
        return jsonify(status="already streaming"), 202

    body = request.get_json(silent=True) or {}
    count = int(body.get("count", request.form.get("count", 20)))
    interval = float(body.get("interval", request.form.get("interval", 0.7)))
    start_consumer()
    _producer_stop.clear()
    _producer_thread = threading.Thread(
        target=_produce, args=(count, interval), name="kafka-producer", daemon=True
    )
    _producer_thread.start()
    return jsonify(status="streaming", count=count, interval=interval)


@stream_bp.post("/stop")
def stop():
    _producer_stop.set()
    return jsonify(status="stopping")


@stream_bp.get("")
def stream():
    """SSE endpoint: emits one event per inferred window."""
    start_consumer()

    def gen():
        q: queue.Queue = queue.Queue(maxsize=200)
        with _subs_lock:
            _subscribers.append(q)
        try:
            yield f"event: hello\ndata: {json.dumps({'recent': list(_recent)})}\n\n"
            while True:
                try:
                    item = q.get(timeout=15)
                    yield f"data: {json.dumps(item)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _subs_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
