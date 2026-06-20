"""
src/ato_detector.py
Core ATO detection logic.

Decision flow:
  ① Extract 33 features
  ② XGBoost → ATO probability
  ③ Compare behavioral anomaly score vs. user baseline
  ④ Decision: ALLOW / OTP_REQUIRED / SUSPEND
  ⑤ Blacklist entities if fraud is confirmed (prob > 0.90)

OTP only – no biometric verification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from feature_extractor import extractor
from live_behavioral_model import live_behavioral_model
from model import ato_model
from utils import (
    add_to_blacklist, is_blacklisted,
    generate_otp, verify_otp,
    setup_logger,
)

setup_logger("ato_detection")

# ─── Risk thresholds ─────────────────────────────────────────────────────────
THRESH_HIGH_PROB       = 0.80
THRESH_MED_PROB        = 0.50
THRESH_HIGH_BEHAVIORAL = 0.30
THRESH_MED_BEHAVIORAL  = 0.20

# In-memory suspended accounts store {account_number: record}
_suspended_accounts: dict[str, dict] = {}
# In-memory recent fraud transactions (last 200)
_fraud_transactions: list[dict] = []
# In-memory recent detections (last 500)
_recent_detections: list[dict] = []


def get_suspended_accounts() -> list[dict]:
    return list(_suspended_accounts.values())


def get_fraud_transactions(limit: int = 50) -> list[dict]:
    return _fraud_transactions[-limit:]


def get_recent_detections(limit: int = 100) -> list[dict]:
    return _recent_detections[-limit:]


class ATODetector:
    """
    Main detector class.  One singleton per process.
    The model is lazy-loaded on the first call to detect_ato().
    """

    def __init__(self) -> None:
        self._model_ready = False

    def _ensure_model(self) -> None:
        if not self._model_ready:
            try:
                ato_model.load()
                self._model_ready = True
            except FileNotFoundError:
                logger.warning("Model file not found – running in demo mode (random probabilities).")

    # ── Main detection ─────────────────────────────────────────────────────
    def detect_ato(
        self,
        transaction: dict[str, Any],
        user_sessions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Full ATO detection pipeline.

        Args:
            transaction:   Dict with transaction fields.
            user_sessions: Past sessions for ATLAS features.

        Returns:
            Structured result dict consumed by the API.
        """
        self._ensure_model()

        account = transaction.get("account_number", "UNKNOWN")
        txn_id  = transaction.get("transaction_id",  "UNKNOWN")

        # ── Step 1: Behavioral anomaly ─────────────────────────────────
        user_baseline = extractor.get_user_baseline(account)
        if user_baseline:
            behavioral_score = extractor.calculate_behavioral_anomaly(transaction, user_baseline)
        else:
            behavioral_score = float(transaction.get("behavioral_anomaly_score", 0.05))

        # ── Step 2: Extract features ───────────────────────────────────
        features = extractor.extract_features(transaction, user_sessions)
        features["behavioral_anomaly_score"] = behavioral_score

        # ── Step 3: ML probability ─────────────────────────────────────
        live_feature_count = 0
        if live_behavioral_model.available:
            metadata = live_behavioral_model.metadata()
            live_features = metadata.get("feature_cols", [])
            live_feature_count = sum(1 for col in live_features if col in transaction)

        if live_behavioral_model.available and live_feature_count >= 8:
            ato_prob = live_behavioral_model.predict_proba(transaction)
        elif self._model_ready:
            ato_prob = ato_model.predict_proba(features)
        else:
            import random, math
            # Demo fallback: derive from risk signals
            risk = (
                features["ip_fraud_score"] * 0.3
                + features["ip_blacklisted"] * 0.15
                + features["new_device"] * 0.10
                + features["new_location"] * 0.10
                + features["beneficiary_blacklisted"] * 0.15
                + min(features["failed_auth_attempts"] / 10, 1) * 0.10
                + behavioral_score * 0.10
            )
            ato_prob = min(0.99, risk + random.uniform(-0.05, 0.05))

        # ── Step 4: Decision logic ─────────────────────────────────────
        high_risk = ato_prob > THRESH_HIGH_PROB or behavioral_score > THRESH_HIGH_BEHAVIORAL
        med_risk  = ato_prob > THRESH_MED_PROB  or behavioral_score > THRESH_MED_BEHAVIORAL

        if high_risk:
            action         = "SUSPEND"
            otp_required   = True
            contact_mobile = True
            suspension_mode = "REVIEW"
            behavioral_warning = (
                f"High behavioral anomaly: {behavioral_score:.3f} "
                f"(threshold: {THRESH_HIGH_BEHAVIORAL})"
            )
            user_capabilities = {
                "view_balance":   True,
                "view_history":   True,
                "call_bank":      True,
                "transfer_money": False,
                "add_beneficiary":False,
                "change_data":    False,
            }
            user_actions = [
                {
                    "action": "SUSPEND_ACCOUNT",
                    "description": "Account suspended for security review (24-48 hrs)",
                    "duration": "24-48 hours",
                    "auto_unlock": True,
                },
                {
                    "action": "TRIGGER_OTP",
                    "description": "OTP sent to registered mobile to verify identity",
                    "otp_method": "SMS",
                },
                {
                    "action": "CONTACT_REGISTERED_MOBILE",
                    "description": (
                        "SMS: 'Suspicious activity detected on your account. "
                        "Verify if this is you.'"
                    ),
                    "mobile": transaction.get("phone_number", "REGISTERED_MOBILE"),
                },
            ]
            # Add to suspended accounts
            _suspended_accounts[account] = {
                "account_number": account,
                "suspension_time": datetime.utcnow().isoformat(),
                "ato_probability": round(ato_prob, 4),
                "behavioral_anomaly_score": round(behavioral_score, 4),
                "status": "SUSPENDED",
                "transaction_id": txn_id,
            }

        elif med_risk:
            action          = "OTP_REQUIRED"
            otp_required    = True
            contact_mobile  = True
            suspension_mode = None
            behavioral_warning = (
                f"Unusual behavioral pattern: {behavioral_score:.3f}"
            )
            user_capabilities = None
            user_actions = [
                {
                    "action": "TRIGGER_OTP",
                    "description": "OTP verification required before transaction proceeds",
                    "otp_method": "SMS",
                },
                {
                    "action": "SEND_WARNING_SMS",
                    "description": (
                        "SMS: 'Unusual activity detected. OTP required to continue.'"
                    ),
                    "mobile": transaction.get("phone_number", "REGISTERED_MOBILE"),
                },
            ]

        else:
            action          = "ALLOW"
            otp_required    = False
            contact_mobile  = False
            suspension_mode = None
            behavioral_warning = None
            user_capabilities = None
            user_actions      = []

        # ── Step 5: OTP generation ─────────────────────────────────────
        otp = None
        if otp_required:
            otp = generate_otp(account)

        # ── Step 6: Conditional blacklisting ──────────────────────────
        ato_confirmed = ato_prob > 0.90 and otp_required
        blacklist_result = self._blacklist_if_confirmed(transaction, ato_confirmed)

        # ── Step 7: Record fraud transaction ──────────────────────────
        if otp_required:
            _fraud_transactions.append({
                "transaction_id":         txn_id,
                "account_number":         account,
                "ato_probability":        round(ato_prob, 4),
                "behavioral_anomaly_score": round(behavioral_score, 4),
                "action":                 action,
                "otp_required":           otp_required,
                "actual_is_ato":          transaction.get("actual_is_ato"),
                "timestamp":              datetime.utcnow().isoformat(),
            })
            # Keep only last 200
            if len(_fraud_transactions) > 200:
                _fraud_transactions.pop(0)

        _recent_detections.append({
            "transaction_id": txn_id,
            "account_number": account,
            "ato_probability": round(ato_prob, 4),
            "behavioral_anomaly_score": round(behavioral_score, 4),
            "action": action,
            "otp_required": otp_required,
            "actual_is_ato": transaction.get("actual_is_ato"),
            "timestamp": datetime.utcnow().isoformat(),
        })
        if len(_recent_detections) > 500:
            _recent_detections.pop(0)

        logger.info(
            f"[ATO] {txn_id} | account={account} | "
            f"prob={ato_prob:.3f} | beh={behavioral_score:.3f} | action={action}"
        )

        return {
            "transaction_id":          txn_id,
            "account_number":          account,
            "ato_probability":         round(ato_prob, 4),
            "behavioral_anomaly_score": round(behavioral_score, 4),
            "action":                  action,
            "suspension_mode":         suspension_mode,
            "otp_required":            otp_required,
            "otp":                     otp,          # in production: send via SMS, don't return in API
            "contact_mobile":          contact_mobile,
            "behavioral_warning":      behavioral_warning,
            "user_capabilities":       user_capabilities,
            "user_actions":            user_actions,
            "blacklist_result":        blacklist_result,
            "timestamp":               datetime.utcnow().isoformat(),
        }

    # ── OTP verification ───────────────────────────────────────────────────
    def verify_transaction_otp(self, account_number: str, otp: str) -> dict[str, Any]:
        """
        Verify OTP submitted by the user.
        - Valid OTP  → unlock account, clear suspension
        - Invalid OTP → keep suspension, escalate
        """
        valid = verify_otp(account_number, otp)

        if valid:
            # Clear suspension
            _suspended_accounts.pop(account_number, None)
            logger.info(f"[OTP] VALID – account {account_number} unlocked")
            return {
                "account_number": account_number,
                "otp_valid":      True,
                "action":         "ACCOUNT_UNLOCKED",
                "message":        "Identity verified. Account unlocked. Transaction can proceed.",
            }
        else:
            logger.warning(f"[OTP] INVALID – account {account_number} remains suspended")
            return {
                "account_number": account_number,
                "otp_valid":      False,
                "action":         "ACCOUNT_BLOCKED",
                "message":        "OTP verification failed. Account fully blocked. KYC re-check required.",
            }

    # ── Blacklisting ───────────────────────────────────────────────────────
    def _blacklist_if_confirmed(
        self,
        transaction: dict[str, Any],
        ato_confirmed: bool,
    ) -> dict[str, Any]:
        if not ato_confirmed:
            return {
                "ip_blacklisted":          False,
                "device_blacklisted":      False,
                "beneficiary_blacklisted": False,
                "reason":                  "Not confirmed as fraud",
            }

        meta = {
            "account":        transaction.get("account_number"),
            "fraud_confirmed":True,
        }

        add_to_blacklist("ip",          str(transaction.get("ip_address",        "")), meta)
        add_to_blacklist("device",      str(transaction.get("device_id",         "")), meta)
        add_to_blacklist("beneficiary", str(transaction.get("beneficiary_account", "")), meta)

        logger.warning(
            f"[BLACKLIST] Confirmed fraud → "
            f"IP={transaction.get('ip_address')} "
            f"Device={transaction.get('device_id')} "
            f"Beneficiary={transaction.get('beneficiary_account')}"
        )
        return {
            "ip_blacklisted":          True,
            "device_blacklisted":      True,
            "beneficiary_blacklisted": True,
            "reason":                  "ATO fraud confirmed",
        }


# Singleton
detector = ATODetector()
