"""scripts/start_live_demo.py

Generate a separate live SMOTE dataset and stream it through Kafka with controlled delay.
This helper is intended for live demo use after the model is already trained.

Steps:
  1. Start the API: python scripts/start_api.py
  2. Start the Kafka consumer: python scripts/start_stream.py
  3. Start the dashboard: python scripts/start_dashboard.py
  4. Run this script to generate and produce live SMOTE transactions.
"""

from __future__ import annotations
from generate_live_stream_dataset import generate_live_stream_dataset
from kafka_live_producer import produce_live_transactions
from utils import DATA_DIR

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a separate live SMOTE dataset and publish it to Kafka."
    )
    parser.add_argument("--rows", type=int, default=5000,
                        help="Base rows before live SMOTE augmentation")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "ato_behavioral_live_stream_smote_50_50.csv",
        help="Path to save the live stream dataset",
    )
    parser.add_argument(
        "--source-output",
        type=Path,
        default=Path(ROOT) / "raw_data" / "ato_behavioral_live_source.csv",
        help="Path to save the base source dataset",
    )
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Delay between transactions in seconds")
    parser.add_argument("--limit", type=int, default=500,
                        help="Maximum number of transactions to stream")
    parser.add_argument("--bootstrap", default="localhost:9092",
                        help="Kafka bootstrap server")
    parser.add_argument("--topic", default="ato_transactions",
                        help="Kafka topic to publish live transactions")
    parser.add_argument("--noise-rate", type=float, default=0.20,
                        help="Noise rate for live source dataset generation")
    parser.add_argument("--label-noise-rate", type=float, default=0.01,
                        help="Label noise for live source dataset generation")
    parser.add_argument("--seed", type=int, default=202,
                        help="Random seed for dataset generation")
    args = parser.parse_args()

    print("Generating separate live SMOTE dataset...")
    live_df = generate_live_stream_dataset(
        rows=args.rows,
        output_path=args.output,
        source_path=args.source_output,
        noise_rate=args.noise_rate,
        label_noise_rate=args.label_noise_rate,
        random_state=args.seed,
    )

    print(f"Saved live stream dataset: {args.output}")
    print(f"Total transactions: {len(live_df):,}")
    print(f"Fraud ratio: {live_df['actual_is_ato'].mean() * 100:.2f}%")
    print("")
    print("Publishing live transactions to Kafka:")
    print(f"  bootstrap={args.bootstrap}")
    print(f"  topic={args.topic}")
    print(f"  delay={args.delay}s")
    print(f"  limit={args.limit}")
    print("")

    produce_live_transactions(
        dataset_path=args.output,
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        delay_sec=args.delay,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
