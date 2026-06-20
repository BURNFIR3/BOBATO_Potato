"""tests/test_ato_detector.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ato_detector import ATODetector
from utils import generate_otp, verify_otp


NORMAL_TXN = {
    "transaction_id":    "TXN-NORMAL-001",
    "account_number":    "BOB11111111",
    "transaction_amount": 5000.0,
    "transaction_type":  "NEFT",
    "ip_address":        "192.168.1.1",
    "ip_fraud_score":    0.05,
    "new_device":        False,
    "new_location":      False,
    "new_beneficiary":   False,
    "failed_auth_attempts": 0,
    "typing_speed_mobile": 4.5,
    "phone_angle_pitch":   45.0,
    "behavioral_anomaly_score": 0.05,
}

FRAUD_TXN = {
    "transaction_id":    "TXN-FRAUD-001",
    "account_number":    "BOB22222222",
    "transaction_amount": 75000.0,
    "transaction_type":  "NEFT",
    "ip_address":        "196.6.8.10",
    "ip_fraud_score":    0.95,
    "ip_blacklisted":    True,
    "device_blacklisted": True,
    "new_device":        True,
    "new_location":      True,
    "new_beneficiary":   True,
    "beneficiary_blacklisted": True,
    "failed_auth_attempts": 5,
    "typing_speed_mobile": 9.0,
    "phone_angle_pitch":   10.0,
    "behavioral_anomaly_score": 0.85,
}


@pytest.fixture
def d():
    return ATODetector()


def test_normal_transaction_low_risk(d):
    result = d.detect_ato(NORMAL_TXN)
    assert result["action"] in ("ALLOW", "OTP_REQUIRED")   # likely ALLOW
    assert 0.0 <= result["ato_probability"] <= 1.0


def test_fraud_transaction_high_risk(d):
    result = d.detect_ato(FRAUD_TXN)
    assert result["action"] in ("OTP_REQUIRED", "SUSPEND")
    assert result["otp_required"] is True


def test_result_has_required_keys(d):
    result = d.detect_ato(NORMAL_TXN)
    required_keys = [
        "transaction_id", "account_number", "ato_probability",
        "behavioral_anomaly_score", "action", "otp_required",
        "blacklist_result", "timestamp",
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_otp_flow():
    otp = generate_otp("BOB99999999")
    assert len(otp) == 6
    assert otp.isdigit()
    assert verify_otp("BOB99999999", otp) is True
    # Second attempt should fail
    assert verify_otp("BOB99999999", otp) is False


def test_blacklist_not_triggered_for_normal(d):
    result = d.detect_ato(NORMAL_TXN)
    bl = result.get("blacklist_result", {})
    assert bl.get("ip_blacklisted") is False
