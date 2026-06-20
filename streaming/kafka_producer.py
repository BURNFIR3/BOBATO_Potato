"""streaming/kafka_producer.py – Produce synthetic transactions to Kafka for testing"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import random
import time
from datetime import datetime
from loguru import logger

from stream import TransactionProducer
from data_processor import _generate_synthetic_dataset


def produce_test_transactions(n: int = 100, delay_sec: float = 0.5) -> None:
    """
    Generate synthetic transactions and push them to the stream.
    Useful for load testing the ATO consumer.
    """
    producer = TransactionProducer()
    df = _generate_synthetic_dataset(n=n)

    logger.info(f"Producing {n} test transactions (delay={delay_sec}s each)…")
    for i, row in df.iterrows():
        txn = row.to_dict()
        txn["transaction_timestamp"] = datetime.utcnow().isoformat()
        success = producer.send_transaction(txn)
        if success:
            label = "🔴 FRAUD" if row["is_ato"] else "✅ NORMAL"
            print(f"[{i+1:4d}/{n}] {label} | Amount: ₹{row['transaction_amount']:,.0f}")
        else:
            print(f"[{i+1:4d}/{n}] ⚠ Send failed (no backend)")
        time.sleep(delay_sec)

    logger.success(f"✓ Produced {n} test transactions.")


if __name__ == "__main__":
    produce_test_transactions(n=50, delay_sec=0.2)
