"""
Runtime loader for the behavioral XGBoost live inference pipeline.

The saved bundle contains the exact preprocessing used during benchmark
training: median imputation, standard scaling, and the XGBoost classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import MODELS_DIR

LIVE_PIPELINE_PATH = MODELS_DIR / "ato_behavioral_live_pipeline.joblib"
LIVE_METADATA_PATH = MODELS_DIR / "ato_behavioral_live_metadata.json"

NON_MODEL_COLUMNS = {
    "fraud_bool",
    "is_ato",
    "actual_is_ato",
    "transaction_id",
    "account_number",
    "transaction_timestamp",
    "transaction_amount",
    "transaction_type",
    "ip_address",
    "ip_country",
    "device_id",
    "beneficiary_account",
    "channel",
    "streamed_at",
}


def numeric_behavioral_features(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding labels and operational metadata."""
    return [
        col for col in df.columns
        if col not in NON_MODEL_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


class BehavioralLiveModel:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return LIVE_PIPELINE_PATH.exists()

    def load(self) -> None:
        if self._bundle is not None:
            return
        if not LIVE_PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Live behavioral pipeline not found: {LIVE_PIPELINE_PATH}")

        import joblib

        self._bundle = joblib.load(LIVE_PIPELINE_PATH)
        if LIVE_METADATA_PATH.exists():
            with open(LIVE_METADATA_PATH) as f:
                self._metadata = json.load(f)

    def predict_proba(self, transaction: dict[str, Any]) -> float:
        self.load()
        assert self._bundle is not None

        feature_cols = self._bundle["feature_cols"]
        pipeline = self._bundle["pipeline"]

        row = {col: transaction.get(col, None) for col in feature_cols}
        X = pd.DataFrame([row])
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return float(pipeline.predict_proba(X[feature_cols])[0, 1])

    def metadata(self) -> dict[str, Any]:
        if not self._metadata and LIVE_METADATA_PATH.exists():
            with open(LIVE_METADATA_PATH) as f:
                self._metadata = json.load(f)
        return self._metadata


live_behavioral_model = BehavioralLiveModel()
