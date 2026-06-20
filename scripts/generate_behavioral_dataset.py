"""Generate the pasted behavioral ATO CSV into raw_data/."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from behavioral_dataset_generator import generate_behavioral_dataset
from utils import RAW_DATA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic behavioral ATO dataset")
    parser.add_argument("--rows", type=int, default=50_000, help="Number of rows to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.08, help="Starting fraud rate before SMOTE")
    parser.add_argument("--noise-rate", type=float, default=0.15, help="Share of each class to make deliberately ambiguous")
    parser.add_argument("--label-noise-rate", type=float, default=0.01, help="Share of labels to flip after feature generation")
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR / "ato_behavioral_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_behavioral_dataset(
        n=args.rows,
        fraud_rate=args.fraud_rate,
        noise_rate=args.noise_rate,
        label_noise_rate=args.label_noise_rate,
        output_path=args.output,
        random_state=args.seed,
    )

    fraud_count = int(df["fraud_bool"].sum())
    print(f"Generated: {args.output}")
    print(f"Rows:      {len(df):,}")
    print(f"Fraud:     {fraud_count:,} ({df['fraud_bool'].mean() * 100:.2f}%)")
    print(f"Noise:     {args.noise_rate * 100:.1f}% ambiguous rows per class")
    print(f"Labels:    {args.label_noise_rate * 100:.1f}% flipped")
