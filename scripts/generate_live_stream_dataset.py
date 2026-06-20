"""Generate a separate SMOTE-balanced dataset for live Kafka streaming."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from behavioral_dataset_generator import generate_behavioral_dataset
from live_behavioral_model import numeric_behavioral_features
from utils import DATA_DIR, RAW_DATA_DIR


def generate_live_stream_dataset(
    rows: int = 5_000,
    output_path: Path = DATA_DIR / "ato_behavioral_live_stream_smote_50_50.csv",
    source_path: Path = RAW_DATA_DIR / "ato_behavioral_live_source.csv",
    noise_rate: float = 0.20,
    label_noise_rate: float = 0.01,
    random_state: int = 202,
) -> pd.DataFrame:
    """Create a live stream dataset that is separate from the trained dataset."""
    source = generate_behavioral_dataset(
        n=rows,
        noise_rate=noise_rate,
        label_noise_rate=label_noise_rate,
        output_path=source_path,
        random_state=random_state,
    )

    feature_cols = numeric_behavioral_features(source)
    X = source[feature_cols]
    y = source["fraud_bool"].astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    fraud_count = int((y == 1).sum())
    k_neighbors = min(5, max(fraud_count - 1, 1))
    smote = SMOTE(sampling_strategy=1.0, k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_imp, y)

    df = pd.DataFrame(X_res, columns=feature_cols)
    df["actual_is_ato"] = np.asarray(y_res, dtype=int)
    df["transaction_id"] = [f"LIVE-{random_state}-{i:06d}" for i in range(len(df))]
    df["account_number"] = [f"BOB{1000000000 + i % 9000000000}" for i in range(len(df))]
    df["transaction_timestamp"] = pd.Timestamp.utcnow().isoformat()
    df["transaction_amount"] = (
        df.get("tx_amount_vs_historical_avg_ratio", pd.Series(1.0, index=df.index)).clip(0.5, 30.0) * 3500
    ).round(2)
    df["transaction_type"] = np.where(df.index % 3 == 0, "IMPS", np.where(df.index % 3 == 1, "UPI", "NEFT"))
    df["ip_address"] = [f"10.{(i // 65536) % 255}.{(i // 256) % 255}.{i % 255}" for i in range(len(df))]
    df["ip_country"] = np.where(df.get("is_foreign_ip", 0).astype(float) > 0.5, "NG", "IN")
    df["device_id"] = [f"DEV-LIVE-{i % 2500:04d}" for i in range(len(df))]
    df["beneficiary_account"] = [f"BEN{7000000000 + i % 900000000}" for i in range(len(df))]
    df["channel"] = "web"
    df["behavioral_anomaly_score"] = df.get("login_hour_anomaly_score", pd.Series(0.05, index=df.index)).clip(0, 1)

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separate live Kafka stream data")
    parser.add_argument("--rows", type=int, default=5_000, help="Base rows before live SMOTE")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "ato_behavioral_live_stream_smote_50_50.csv")
    parser.add_argument("--source-output", type=Path, default=RAW_DATA_DIR / "ato_behavioral_live_source.csv")
    parser.add_argument("--noise-rate", type=float, default=0.20)
    parser.add_argument("--label-noise-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()

    live_df = generate_live_stream_dataset(
        rows=args.rows,
        output_path=args.output,
        source_path=args.source_output,
        noise_rate=args.noise_rate,
        label_noise_rate=args.label_noise_rate,
        random_state=args.seed,
    )
    fraud = int(live_df["actual_is_ato"].sum())
    print(f"Live stream dataset: {args.output}")
    print(f"Source dataset:      {args.source_output}")
    print(f"Rows:                {len(live_df):,}")
    print(f"Fraud:               {fraud:,} ({live_df['actual_is_ato'].mean() * 100:.2f}%)")
