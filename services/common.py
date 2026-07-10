"""Shared Kafka plumbing and wire-format helpers for PHAROS Phase 1.

The wire format (see ``docs/wire_format.md``) is versioned JSON. Producers emit
*pre-normalization*, model-ready-shape records; the scorer owns normalization
so there is a single source of truth for the transform.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, Iterator

from confluent_kafka import Consumer, KafkaError, Producer

BOOTSTRAP_DEFAULT = "localhost:9092"

TOPIC_PHYSICS = "events.physics"
TOPIC_PDM = "events.pdm"
TOPIC_SCOUTING = "anomalies.scouting"
TOPIC_ALERTS = "alerts.pdm"

SCHEMA_PHYSICS = "pharos.physics.v1"
SCHEMA_PDM = "pharos.pdm.v1"
SCHEMA_SCOUTING = "pharos.scouting.v1"
SCHEMA_ALERT = "pharos.pdm_alert.v1"


def now_ns() -> int:
    """Wall-clock nanoseconds; producer/scorer run on the same host so
    producer-ts -> scorer-ts differences are meaningful latencies."""
    return time.time_ns()


def new_event_id(prefix: str, seq: int) -> str:
    return f"{prefix}-{seq:08d}-{uuid.uuid4().hex[:8]}"


def make_producer(bootstrap: str = BOOTSTRAP_DEFAULT) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap,
        "linger.ms": 5,
        "compression.type": "lz4",
        "message.max.bytes": 2_000_000,
    })


def make_consumer(topic: str, group: str,
                  bootstrap: str = BOOTSTRAP_DEFAULT,
                  from_beginning: bool = True) -> Consumer:
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest" if from_beginning else "latest",
        "enable.auto.commit": True,
        "fetch.message.max.bytes": 2_000_000,
    })
    consumer.subscribe([topic])
    return consumer


def produce_json(producer: Producer, topic: str, record: Dict[str, Any],
                 key: str | None = None) -> None:
    producer.produce(topic, json.dumps(record).encode("utf-8"),
                     key=key.encode("utf-8") if key else None)


def consume_json(consumer: Consumer, idle_timeout_s: float = 10.0,
                 poll_s: float = 1.0) -> Iterator[Dict[str, Any]]:
    """Yield decoded JSON records; stop after ``idle_timeout_s`` with no data."""
    idle = 0.0
    while True:
        msg = consumer.poll(poll_s)
        if msg is None:
            idle += poll_s
            if idle >= idle_timeout_s:
                return
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise RuntimeError(f"Kafka error: {msg.error()}")
        idle = 0.0
        yield json.loads(msg.value().decode("utf-8"))


class RateLimiter:
    """Pace an event loop to ``rate`` events/sec (0 or None = unthrottled)."""

    def __init__(self, rate: float | None) -> None:
        self.interval = 1.0 / rate if rate else 0.0
        self._next = time.perf_counter()

    def wait(self) -> None:
        if not self.interval:
            return
        self._next += self.interval
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            # Fell behind; don't accumulate debt.
            self._next = time.perf_counter()
