"""
src/smote_augmentation.py
SMOTE over-sampling to address class imbalance in ATO dataset.
Target: 50% fraud ratio after augmentation.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from loguru import logger

from utils import DATA_DIR, setup_logger

setup_logger("ato_detection")

FEATURE_COLS = [
    "ip_fraud_score", "ip_blacklisted", "device_blacklisted", "new_device",
    "new_location", "location_mismatch_ip", "new_beneficiary", "beneficiary_blacklisted",
    "failed_auth_attempts", "session_age_minutes",
    "txn_frequency_1h", "txn_frequency_24h", "amount_vs_avg", "amount_percentile",
    "account_age_days", "previous_fraud_flags",
    "typing_speed_mobile", "typing_rhythm_mobile",
    "phone_angle_pitch", "phone_angle_roll", "swipe_speed_mobile",
    "mouse_path_complexity", "typing_speed_web", "typing_rhythm_web",
    "mouse_speed_web", "scroll_velocity_mobile", "scroll_velocity_web",
    "session_duration", "behavioral_anomaly_score",
]

METADATA_COLS = [
    "transaction_id", "account_number", "transaction_timestamp",
    "transaction_amount", "transaction_type", "ip_address",
    "ip_country", "ip_city", "ip_blacklisted", "ip_fraud_score",
    "device_id", "device_type", "device_os", "device_blacklisted", "new_device",
    "location_city", "location_state", "new_location", "location_mismatch_ip",
    "auth_method", "failed_auth_attempts", "session_age_minutes",
    "beneficiary_account", "beneficiary_name", "new_beneficiary", "beneficiary_blacklisted",
    "channel", "txn_frequency_1h", "txn_frequency_24h",
    "amount_vs_avg", "amount_percentile",
    "account_age_days", "previous_fraud_flags", "kyc_status", "risk_profile",
]


def augment_with_smote(
    dataset_path: str = str(DATA_DIR / "ato_dataset_processed.csv"),
    sampling_strategy: float = 1.0,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply SMOTE to balance the ATO dataset.

    Args:
        dataset_path:      Path to processed dataset.
        sampling_strategy: Desired minority:majority ratio (1.0 -> 50/50).
        k_neighbors:       SMOTE k-neighbors.
        random_state:      Reproducibility seed.

    Returns:
        Augmented DataFrame saved to data/ato_dataset_with_smote.csv
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.error("imbalanced-learn not installed. Run: pip install imbalanced-learn")
        raise

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df):,} records from {dataset_path}")

    # Ensure feature cols exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        logger.warning(f"Missing feature columns (will use 0): {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[FEATURE_COLS].astype(float)
    y = df["is_ato"].astype(int)

    n_normal = int((y == 0).sum())
    n_fraud  = int((y == 1).sum())
    logger.info(f"Before SMOTE → Normal: {n_normal:,} | ATO: {n_fraud:,} ({n_fraud/len(y)*100:.1f}%)")

    if n_fraud < k_neighbors + 1:
        logger.error(f"Too few fraud samples ({n_fraud}) for SMOTE with k={k_neighbors}.")
        return df

    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)

    df_aug = pd.DataFrame(X_res, columns=FEATURE_COLS)
    df_aug["is_ato"] = y_res.astype(int)

    n_total = len(df_aug)
    rng = np.random.default_rng(random_state)

    # Re-attach metadata columns by sampling from originals
    meta_df = df[METADATA_COLS] if all(c in df.columns for c in METADATA_COLS) else pd.DataFrame()
    if not meta_df.empty:
        sampled_meta = meta_df.sample(n=n_total, replace=True, random_state=random_state).reset_index(drop=True)
        for col in METADATA_COLS:
            if col in df_aug.columns:
                continue
            df_aug[col] = sampled_meta[col]

    # Generate unique transaction IDs
    df_aug["transaction_id"] = [
        f"TXN_SMOTE{i:07d}" for i in range(n_total)
    ]

    out_path = DATA_DIR / "ato_dataset_with_smote.csv"
    df_aug.to_csv(out_path, index=False)

    n_fraud_new  = int((y_res == 1).sum())
    n_normal_new = int((y_res == 0).sum())
    logger.success(f"✓ Augmented dataset saved → {out_path}")
    logger.info(f"  After SMOTE → Normal: {n_normal_new:,} | ATO: {n_fraud_new:,} ({n_fraud_new/len(y_res)*100:.1f}%)")

    return df_aug


if __name__ == "__main__":
    augment_with_smote()
