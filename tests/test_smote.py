"""tests/test_smote.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
from smote_augmentation import augment_with_smote, FEATURE_COLS
from data_processor import _generate_synthetic_dataset, DATA_DIR


def _ensure_processed_dataset():
    """Create a processed dataset if it doesn't exist yet."""
    path = DATA_DIR / "ato_dataset_processed.csv"
    if not path.exists():
        df = _generate_synthetic_dataset(n=500)
        df.to_csv(path, index=False)
    return str(path)


def test_smote_output_has_more_fraud():
    path = _ensure_processed_dataset()
    df_orig = pd.read_csv(path)
    orig_fraud_pct = df_orig["is_ato"].mean()

    df_aug = augment_with_smote(dataset_path=path, sampling_strategy=0.4)
    aug_fraud_pct = df_aug["is_ato"].mean()

    assert aug_fraud_pct > orig_fraud_pct, "SMOTE should increase fraud ratio"
    assert aug_fraud_pct > 0.25, "Fraud ratio should be > 25% after SMOTE"


def test_smote_feature_columns_preserved():
    path = _ensure_processed_dataset()
    df_aug = augment_with_smote(dataset_path=path)
    for col in FEATURE_COLS:
        assert col in df_aug.columns, f"Feature column missing: {col}"
