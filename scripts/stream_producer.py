"""Stream signal windows to Kafka to simulate a live sensor feed (v3).

    python scripts/stream_producer.py [count] [interval_seconds]

Publishes random labeled windows to the Kafka topic; the API's background
consumer runs inference and pushes results to the Live feed page over SSE.
Env: KAFKA_BOOTSTRAP (default localhost:9092), KAFKA_TOPIC (default vibration).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "infra" / "api"
for _p in (str(PROJECT_ROOT), str(API_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from kafka import KafkaProducer  # noqa: E402

import engine  # noqa: E402
from src import synthetic  # noqa: E402

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "vibration")


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

    engine.load_engine()
    eng = engine.get_engine()
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    rng = np.random.default_rng()
    print(f"Streaming {count} windows to {BOOTSTRAP} topic '{TOPIC}' ...")
    for i in range(count):
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
        print(f"  sent {i + 1}/{count} ({eng.conditions[idx]})")
        time.sleep(interval)
    producer.flush()
    producer.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
