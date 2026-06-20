"""Produce live demo transactions to Kafka with configurable delays.

This producer uses a separate SMOTE-generated live dataset that is distinct from
training data. Stream transactions through the main ATO Kafka topic so the
real-time consumer can score them and the dashboard can display live results.
"""

from __future__ import annotations
from utils import DATA_DIR

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _clean_payload(row: dict) -> dict:
    payload = {}
    for key, value in row.items():
        if pd.isna(value):
            payload[key] = None
        elif hasattr(value, "item"):
            payload[key] = value.item()
        else:
            payload[key] = value
    return payload


def produce_live_transactions(
    dataset_path: Path,
    bootstrap_servers: str,
    topic: str,
    delay_sec: float,
    limit: int | None,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Live dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if limit:
        df = df.head(limit)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        retries=3,
    )

    print(
        f"Kafka producer connected: topic={topic}, bootstrap={bootstrap_servers}")
    print(f"Streaming {len(df):,} transactions with {delay_sec:.2f}s delay")

    for idx, row in df.iterrows():
        payload = _clean_payload(row.to_dict())
        producer.send(topic, value=payload)
        producer.flush()
        print(
            f"{idx + 1:05d}/{len(df):05d} | "
            f"txn={payload.get('transaction_id')} | actual={payload.get('actual_is_ato')}"
        )
        time.sleep(delay_sec)

    producer.close()
    print("Kafka stream completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce live ATO transactions to Kafka")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR /
                        "ato_behavioral_live_stream_smote_50_50.csv")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="ato_transactions")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    produce_live_transactions(
        dataset_path=args.dataset,
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        delay_sec=args.delay,
        limit=args.limit,
    )
