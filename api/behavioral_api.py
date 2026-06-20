"""
api/behavioral_api.py
FastAPI endpoints for user behavioral baseline management.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from feature_extractor import extractor

router = APIRouter(prefix="/api/v1/behavioral", tags=["Behavioral Biometrics"])


class SessionData(BaseModel):
    typing_speed_mobile:    float = 4.5
    typing_rhythm_mobile:   float = 0.15
    phone_angle_pitch:      float = 45.0
    phone_angle_roll:       float = 10.0
    swipe_speed_mobile:     float = 0.5
    mouse_path_complexity:  float = 0.3
    session_duration:       float = 120.0


class UpdateBaselineRequest(BaseModel):
    sessions: List[SessionData]


@router.post("/baseline/{account_number}", summary="Update user behavioral baseline")
def update_baseline(account_number: str, req: UpdateBaselineRequest) -> dict:
    if len(req.sessions) < 3:
        raise HTTPException(status_code=400, detail="Minimum 3 sessions required to build baseline")
    sessions_dicts = [s.model_dump() for s in req.sessions]
    baseline = extractor.create_user_baseline(account_number, sessions_dicts)
    return {
        "account_number": account_number,
        "baseline":       baseline,
        "sessions_used":  len(sessions_dicts),
        "status":         "Baseline updated",
    }


@router.get("/baseline/{account_number}", summary="Get user behavioral baseline")
def get_baseline(account_number: str) -> dict:
    baseline = extractor.get_user_baseline(account_number)
    if not baseline:
        raise HTTPException(status_code=404, detail=f"No baseline found for {account_number}")
    return {"account_number": account_number, "baseline": baseline}


@router.get("/baselines", summary="List all users with behavioral baselines")
def list_baselines() -> dict:
    return {
        "count":    len(extractor.user_baselines),
        "accounts": list(extractor.user_baselines.keys()),
    }


class AnomalyCheckRequest(BaseModel):
    account_number: str
    session:        SessionData


@router.post("/anomaly-check", summary="Check behavioral anomaly for a session")
def check_anomaly(req: AnomalyCheckRequest) -> dict:
    baseline = extractor.get_user_baseline(req.account_number)
    if not baseline:
        return {
            "account_number":         req.account_number,
            "behavioral_anomaly_score": 0.05,
            "has_baseline":           False,
            "note":                   "No baseline – using default score",
        }
    score = extractor.calculate_behavioral_anomaly(req.session.model_dump(), baseline)
    return {
        "account_number":         req.account_number,
        "behavioral_anomaly_score": round(score, 4),
        "has_baseline":           True,
        "risk_level":             "HIGH" if score > 0.30 else ("MEDIUM" if score > 0.20 else "LOW"),
    }
