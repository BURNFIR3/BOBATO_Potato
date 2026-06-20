"""
src/data_processor.py
Process raw CSV datasets from raw_data/ folder.
Handles schema normalization, type coercion, and saves to data/ato_dataset_processed.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from utils import RAW_DATA_DIR, DATA_DIR, setup_logger

setup_logger("ato_detection")

# ─── Features from Base.csv dataset ───────────────────────────────────────────
# All 31 features + fraud_bool target
NUMERIC_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
    "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "credit_risk_score",
    "email_is_free", "phone_home_valid", "phone_mobile_valid", "bank_months_count",
    "has_other_cards", "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w",
    "device_fraud_count", "month",
]

CATEGORICAL_FEATURES = [
    "payment_type", "employment_status", "housing_status", "source", "device_os"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist."""
    for col in ALL_FEATURES:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing from dataset")
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric features to float, handle categorical."""
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # LabelEncode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def process_raw_dataset() -> pd.DataFrame | None:
    """
    Load and process all CSV files in raw_data/.
    Handles fraud_bool → is_ato target renaming.
    Returns the processed DataFrame.
    """
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        logger.error("No CSV files found in raw_data/")
        return None

    logger.info(f"Found {len(csv_files)} CSV file(s) in raw_data/")
    frames = []

    for fp in csv_files:
        logger.info(f"  Reading: {fp.name}")
        try:
            df_temp = pd.read_csv(fp, low_memory=False)
            frames.append(df_temp)
        except Exception as e:
            logger.error(f"  ❌ Failed to read {fp.name}: {e}")

    if not frames:
        logger.error("All CSV reads failed.")
        return None

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df):,} total records")

    # Rename target column if needed
    if "fraud_bool" in df.columns and "is_ato" not in df.columns:
        df.rename(columns={"fraud_bool": "is_ato"}, inplace=True)

    # Select only available features + target
    available_cols = [c for c in ALL_FEATURES if c in df.columns] + ["is_ato"]
    df = df[available_cols].copy()

    # Type coercion
    df = _coerce_types(df)

    # Drop rows with missing target
    df.dropna(subset=["is_ato"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save
    out_path = DATA_DIR / "ato_dataset_processed.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    ato_count = int(df["is_ato"].sum())
    pct = ato_count / len(df) * 100 if len(df) > 0 else 0
    logger.success(f"✓ Processed dataset saved → {out_path}")
    logger.info(
        f"  Total records: {len(df):,} | Fraud: {ato_count:,} ({pct:.2f}%)")
    return df


if __name__ == "__main__":
    process_raw_dataset()
