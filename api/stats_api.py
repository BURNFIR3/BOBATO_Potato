"""
api/stats_api.py
System statistics API – consumed by the Streamlit dashboard.
"""

from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from fastapi import APIRouter

from utils import load_blacklist, MODELS_DIR
from ato_detector import (
    get_fraud_transactions,
    get_recent_detections,
    get_suspended_accounts,
    detector,
)
from live_behavioral_model import live_behavioral_model

router = APIRouter(prefix="/api/v1", tags=["Statistics"])


@router.get("/ato-stats", summary="System-level ATO statistics")
def ato_stats() -> dict:
    # Load model metrics if available
    behavioral_metrics_path = MODELS_DIR / "behavioral_benchmark_metrics.json"
    metrics_path = behavioral_metrics_path if behavioral_metrics_path.exists() else MODELS_DIR / "training_metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    bl_summary = {
        "blacklisted_ips":          len(load_blacklist("ip")),
        "blacklisted_devices":      len(load_blacklist("device")),
        "blacklisted_beneficiaries":len(load_blacklist("beneficiary")),
    }
    fraud_txns    = get_fraud_transactions()
    recent        = get_recent_detections()
    suspended     = get_suspended_accounts()
    action_counts: dict[str, int] = {}
    for item in recent:
        action = item.get("action", "UNKNOWN")
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "service":               "ATO Detection System",
        "status":                "Online",
        "model_loaded":          detector._model_ready,
        "model_accuracy":        f"{metrics.get('accuracy', 0)*100:.1f}%",
        "model_auc":             f"{metrics.get('auc_roc', 0):.4f}",
        "model_f1":              f"{metrics.get('f1_score', 0):.4f}",
        "false_positive_rate":   f"{metrics.get('false_positive_rate', 0)*100:.2f}%",
        "fraud_transactions_count": len(fraud_txns),
        "suspended_accounts_count": len(suspended),
        "recent_detections_count": len(recent),
        "action_counts": action_counts,
        "live_behavioral_model_loaded": live_behavioral_model.available,
        **bl_summary,
    }


@router.get("/fraud-transactions", summary="Recent fraud transactions (last 50)")
def fraud_transactions() -> list:
    return get_fraud_transactions(50)


@router.get("/recent-detections", summary="Recent live detections")
def recent_detections() -> list:
    return get_recent_detections(100)


@router.get("/suspended-accounts", summary="Currently suspended accounts")
def suspended_accounts() -> list:
    return get_suspended_accounts()


@router.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok", "service": "BOB ATO Detection API"}
