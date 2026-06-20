"""tests/test_data_processor.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
from data_processor import (
    _generate_synthetic_dataset,
    _add_missing_columns,
    _coerce_types,
    process_raw_dataset,
)


def test_synthetic_dataset_shape():
    df = _generate_synthetic_dataset(n=200)
    assert len(df) == 200
    assert "is_ato" in df.columns
    assert "transaction_id" in df.columns


def test_synthetic_dataset_fraud_ratio():
    df = _generate_synthetic_dataset(n=1000)
    fraud_pct = df["is_ato"].mean()
    # Should be roughly 10% ± 5%
    assert 0.04 < fraud_pct < 0.20


def test_add_missing_columns():
    df = pd.DataFrame({"transaction_id": ["TXN001"], "is_ato": [0]})
    df = _add_missing_columns(df)
    assert "ip_fraud_score" in df.columns
    assert "behavioral_anomaly_score" in df.columns


def test_coerce_types():
    df = pd.DataFrame({
        "ip_blacklisted": ["true", "false", "1"],
        "ip_fraud_score": ["0.5", "abc", None],
    })
    df = _coerce_types(df)
    assert df["ip_blacklisted"].tolist() == [1, 0, 1]
    assert df["ip_fraud_score"].tolist() == [0.5, 0.0, 0.0]


def test_process_raw_dataset_runs():
    df = process_raw_dataset()
    assert df is not None
    assert len(df) > 0
    assert "is_ato" in df.columns
