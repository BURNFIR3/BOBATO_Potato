"""
api/ato_api.py
FastAPI endpoint: POST /api/v1/detect-ato
                  POST /api/v1/verify-otp
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ato_detector import detector

router = APIRouter(prefix="/api/v1", tags=["ATO Detection"])


# ─── Request / Response Models ────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    transaction_id:        Optional[str]   = Field(None,    description="Unique transaction ID")
    account_number:        str             = Field(...,     description="Bank account number")
    transaction_amount:    float           = Field(...,     description="Amount in INR")
    transaction_type:      str             = Field("NEFT",  description="NEFT/IMPS/UPI/RTGS")
    ip_address:            str             = Field("0.0.0.0")
    ip_country:            Optional[str]   = "IN"
    ip_blacklisted:        bool            = False
    ip_fraud_score:        float           = Field(0.0, ge=0, le=1)
    device_id:             Optional[str]   = None
    device_type:           Optional[str]   = "mobile"
    device_os:             Optional[str]   = None
    device_blacklisted:    bool            = False
    new_device:            bool            = False
    location_city:         Optional[str]   = None
    location_state:        Optional[str]   = None
    new_location:          bool            = False
    location_mismatch_ip:  bool            = False
    auth_method:           str             = "PASSWORD"
    failed_auth_attempts:  int             = Field(0, ge=0)
    session_age_minutes:   int             = Field(30, ge=0)
    beneficiary_account:   Optional[str]   = None
    beneficiary_name:      Optional[str]   = None
    new_beneficiary:       bool            = False
    beneficiary_blacklisted: bool          = False
    channel:               str             = "mobile"
    txn_frequency_1h:      int             = Field(1, ge=0)
    txn_frequency_24h:     int             = Field(3, ge=0)
    amount_vs_avg:         float           = Field(1.0, ge=0)
    amount_percentile:     float           = Field(0.5, ge=0, le=1)
    account_age_days:      int             = Field(365, ge=0)
    previous_fraud_flags:  int             = Field(0, ge=0)
    kyc_status:            str             = "VERIFIED"
    risk_profile:          str             = "LOW"
    phone_number:          Optional[str]   = None
    # Behavioral
    typing_speed_mobile:    float          = 4.5
    typing_rhythm_mobile:   float          = 0.15
    phone_angle_pitch:      float          = 45.0
    phone_angle_roll:       float          = 10.0
    swipe_speed_mobile:     float          = 0.5
    mouse_path_complexity:  float          = 0.3
    typing_speed_web:       float          = 5.0
    typing_rhythm_web:      float          = 0.12
    mouse_speed_web:        float          = 300.0
    scroll_velocity_mobile: float          = 0.6
    scroll_velocity_web:    float          = 0.4
    session_duration:       float          = 120.0
    behavioral_anomaly_score: float        = 0.05
    # ATLAS
    past_sessions:          Optional[list] = None


class OTPVerifyRequest(BaseModel):
    account_number: str
    otp:            str = Field(..., min_length=4, max_length=8)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/detect-ato", summary="Detect ATO for a transaction")
def detect_ato(request: TransactionRequest) -> dict:
    import uuid
    txn = request.model_dump()
    if not txn.get("transaction_id"):
        txn["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12].upper()}"
    result = detector.detect_ato(txn, user_sessions=request.past_sessions)
    # Remove raw OTP from API response (would be sent via SMS in production)
    result.pop("otp", None)
    return result


@router.post("/verify-otp", summary="Verify OTP submitted by user")
def verify_otp_endpoint(request: OTPVerifyRequest) -> dict:
    return detector.verify_transaction_otp(request.account_number, request.otp)
