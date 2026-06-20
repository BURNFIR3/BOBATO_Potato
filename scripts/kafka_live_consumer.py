"""Consume live Kafka transactions and submit them to the ATO API."""

from __future__ import annotations

import argparse
import json
import time

import requests
from kafka import KafkaConsumer


def consume_to_api(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    api_url: str,
) -> None:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    print(
        f"Kafka consumer connected: topic={topic}, bootstrap={bootstrap_servers}")
    print(f"Forwarding messages to: {api_url}")

    for msg in consumer:
        payload = msg.value
        try:
            response = requests.post(api_url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            print(
                f"{result.get('transaction_id')} | "
                f"action={result.get('action')} | "
                f"prob={result.get('ato_probability'):.4f} | "
                f"actual={payload.get('actual_is_ato')}"
            )
        except Exception as exc:
            print(f"Failed to process Kafka message: {exc}")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kafka consumer that calls the ATO API")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="ato_transactions")
    parser.add_argument("--group-id", default="ato_live_api_consumer")
    parser.add_argument(
        "--api-url", default="http://localhost:8000/api/v1/detect-ato")
    args = parser.parse_args()

    consume_to_api(
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        group_id=args.group_id,
        api_url=args.api_url,
    )
