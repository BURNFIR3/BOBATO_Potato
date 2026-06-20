"""
src/feature_extractor.py
Feature selection for XGBoost model based on actual dataset schema.
"""

from __future__ import annotations

from loguru import logger
from utils import setup_logger

setup_logger("ato_detection")

# ─── Feature columns from Base.csv dataset ───────────────────────────────────
# These are the actual features that the XGBoost model uses for prediction
MODEL_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
    "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "credit_risk_score",
    "email_is_free", "phone_home_valid", "phone_mobile_valid", "bank_months_count",
    "has_other_cards", "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w",
    "device_fraud_count", "month", "payment_type", "employment_status",
    "housing_status", "source", "device_os",
]

class FeatureExtractor:
    def get_user_baseline(self, account: str) -> dict | None:
        return None

    def calculate_behavioral_anomaly(self, transaction: dict, user_baseline: dict | None) -> float:
        # Default to 0.05 since Base.csv does not have behavioral metrics
        return float(transaction.get("behavioral_anomaly_score", 0.05))

    def extract_features(self, transaction: dict, user_sessions: list[dict] | None = None) -> dict[str, float]:
        features = {}
        for col in MODEL_FEATURES:
            val = transaction.get(col, 0.0)
            try:
                features[col] = float(val) if val is not None else 0.0
            except ValueError:
                features[col] = 0.0
        return features

extractor = FeatureExtractor()

