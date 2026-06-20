"""
src/stream.py
Real-time transaction streaming using Kafka (primary) or Redis (fallback).
Consumes transactions, runs ATO detection, publishes alerts.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any, Callable

from loguru import logger
from utils import setup_logger

setup_logger("ato_detection")

_KAFKA_AVAILABLE = False
_REDIS_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    _KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("kafka-python not installed – Kafka streaming unavailable.")

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    logger.warning("redis not installed – Redis streaming unavailable.")


# ─── Kafka helpers ────────────────────────────────────────────────────────────

def _make_kafka_producer(bootstrap_servers: str = "localhost:9092") -> Any:
    if not _KAFKA_AVAILABLE:
        raise RuntimeError("kafka-python not installed.")
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        retries=3,
    )


def _make_kafka_consumer(
    topic: str = "ato_transactions",
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "ato_consumer_group",
) -> Any:
    if not _KAFKA_AVAILABLE:
        raise RuntimeError("kafka-python not installed.")
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )


# ─── Redis helpers ────────────────────────────────────────────────────────────

def _make_redis_client(host: str = "localhost", port: int = 6379, db: int = 0) -> Any:
    if not _REDIS_AVAILABLE:
        raise RuntimeError("redis not installed.")
    return redis_lib.Redis(host=host, port=port, db=db, decode_responses=True)


# ─── Streaming classes ────────────────────────────────────────────────────────

class TransactionProducer:
    """Push transactions to Kafka (or Redis as fallback)."""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self._kafka_producer = None
        self._redis_client   = None

        if _KAFKA_AVAILABLE:
            try:
                self._kafka_producer = _make_kafka_producer(bootstrap_servers)
                logger.info("Kafka producer connected.")
            except Exception as e:
                logger.warning(f"Kafka connection failed: {e}")

        if self._kafka_producer is None and _REDIS_AVAILABLE:
            try:
                self._redis_client = _make_redis_client()
                logger.info("Redis fallback producer connected.")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")

    def send_transaction(self, transaction: dict, topic: str = "ato_transactions") -> bool:
        """Send a transaction dict to the stream."""
        payload = {**transaction, "streamed_at": datetime.utcnow().isoformat()}

        if self._kafka_producer:
            try:
                self._kafka_producer.send(topic, value=payload)
                self._kafka_producer.flush()
                return True
            except Exception as e:
                logger.error(f"Kafka send failed: {e}")

        if self._redis_client:
            try:
                self._redis_client.lpush(topic, json.dumps(payload, default=str))
                self._redis_client.ltrim(topic, 0, 9999)   # keep last 10k
                return True
            except Exception as e:
                logger.error(f"Redis push failed: {e}")

        logger.warning("No streaming backend available – transaction not streamed.")
        return False


class ATOConsumer:
    """
    Consume transactions from Kafka / Redis and run ATO detection.
    Runs in a background thread.
    """

    def __init__(
        self,
        on_alert: Callable[[dict], None] | None = None,
        bootstrap_servers: str = "localhost:9092",
        transaction_topic: str = "ato_transactions",
        alert_topic:       str = "ato_alerts",
    ):
        self._on_alert   = on_alert or (lambda x: logger.info(f"[ALERT] {x}"))
        self._topic      = transaction_topic
        self._alert_topic = alert_topic
        self._running    = False
        self._thread: threading.Thread | None = None
        self._bootstrap  = bootstrap_servers

        self._kafka_consumer  = None
        self._kafka_producer  = None
        self._redis_client    = None

    def start(self) -> None:
        """Start the consumer in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ATOConsumer")
        self._thread.start()
        logger.info("ATO Consumer started.")

    def stop(self) -> None:
        self._running = False
        logger.info("ATO Consumer stopped.")

    def _run(self) -> None:
        from ato_detector import detector

        if _KAFKA_AVAILABLE:
            try:
                self._kafka_consumer = _make_kafka_consumer(self._topic, self._bootstrap)
                self._kafka_producer = _make_kafka_producer(self._bootstrap)
                logger.info("Kafka consumer connected.")
                self._consume_kafka(detector)
                return
            except Exception as e:
                logger.warning(f"Kafka consumer failed: {e}")

        if _REDIS_AVAILABLE:
            try:
                self._redis_client = _make_redis_client()
                logger.info("Redis consumer connected.")
                self._consume_redis(detector)
                return
            except Exception as e:
                logger.warning(f"Redis consumer failed: {e}")

        logger.error("No streaming backend – consumer exiting.")

    def _consume_kafka(self, detector) -> None:
        for msg in self._kafka_consumer:
            if not self._running:
                break
            txn = msg.value
            self._process(txn, detector)

    def _consume_redis(self, detector) -> None:
        while self._running:
            raw = self._redis_client.brpop(self._topic, timeout=1)
            if raw:
                _, payload = raw
                txn = json.loads(payload)
                self._process(txn, detector)

    def _process(self, txn: dict, detector) -> None:
        try:
            result = detector.detect_ato(txn)
            if result["action"] != "ALLOW":
                self._on_alert(result)
                if self._kafka_producer:
                    self._kafka_producer.send(self._alert_topic, value=result)
                elif self._redis_client:
                    self._redis_client.lpush(
                        self._alert_topic,
                        json.dumps(result, default=str),
                    )
        except Exception as e:
            logger.error(f"Error processing transaction {txn.get('transaction_id')}: {e}")


# ─── Convenience start functions ──────────────────────────────────────────────

def start_consumer(on_alert: Callable | None = None) -> ATOConsumer:
    consumer = ATOConsumer(on_alert=on_alert)
    consumer.start()
    return consumer
