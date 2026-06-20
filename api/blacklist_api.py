"""
api/blacklist_api.py
FastAPI endpoints for IP / Device / Beneficiary blacklist management.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils import load_blacklist, save_blacklist, add_to_blacklist, is_blacklisted, BLACKLISTS_DIR

router = APIRouter(prefix="/api/v1/blacklist", tags=["Blacklists"])


# ─── Models ───────────────────────────────────────────────────────────────────

class BlacklistAddRequest(BaseModel):
    value:   str
    reason:  Optional[str] = "Manual addition"
    account: Optional[str] = None


# ─── IP ───────────────────────────────────────────────────────────────────────

@router.get("/ip", summary="Get IP blacklist")
def get_ip_blacklist() -> dict:
    return load_blacklist("ip")


@router.post("/ip", summary="Add IP to blacklist")
def add_ip(req: BlacklistAddRequest) -> dict:
    add_to_blacklist("ip", req.value, {"reason": req.reason, "account": req.account, "fraud_confirmed": True})
    return {"message": f"IP {req.value} added to blacklist"}


@router.delete("/ip/{ip}", summary="Remove IP from blacklist")
def remove_ip(ip: str) -> dict:
    bl = load_blacklist("ip")
    if ip not in bl:
        raise HTTPException(status_code=404, detail=f"IP {ip} not in blacklist")
    del bl[ip]
    save_blacklist("ip", bl)
    return {"message": f"IP {ip} removed from blacklist"}


@router.get("/ip/{ip}/check", summary="Check if IP is blacklisted")
def check_ip(ip: str) -> dict:
    return {"ip": ip, "blacklisted": is_blacklisted("ip", ip)}


# ─── Device ───────────────────────────────────────────────────────────────────

@router.get("/device", summary="Get device blacklist")
def get_device_blacklist() -> dict:
    return load_blacklist("device")


@router.post("/device", summary="Add device to blacklist")
def add_device(req: BlacklistAddRequest) -> dict:
    add_to_blacklist("device", req.value, {"reason": req.reason, "account": req.account, "fraud_confirmed": True})
    return {"message": f"Device {req.value} added to blacklist"}


@router.delete("/device/{device_id}", summary="Remove device from blacklist")
def remove_device(device_id: str) -> dict:
    bl = load_blacklist("device")
    if device_id not in bl:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not in blacklist")
    del bl[device_id]
    save_blacklist("device", bl)
    return {"message": f"Device {device_id} removed from blacklist"}


# ─── Beneficiary ──────────────────────────────────────────────────────────────

@router.get("/beneficiary", summary="Get beneficiary blacklist")
def get_beneficiary_blacklist() -> dict:
    return load_blacklist("beneficiary")


@router.post("/beneficiary", summary="Add beneficiary to blacklist")
def add_beneficiary(req: BlacklistAddRequest) -> dict:
    add_to_blacklist("beneficiary", req.value, {"reason": req.reason, "account": req.account, "fraud_confirmed": True})
    return {"message": f"Beneficiary {req.value} added to blacklist"}


@router.delete("/beneficiary/{acc}", summary="Remove beneficiary from blacklist")
def remove_beneficiary(acc: str) -> dict:
    bl = load_blacklist("beneficiary")
    if acc not in bl:
        raise HTTPException(status_code=404, detail=f"Beneficiary {acc} not in blacklist")
    del bl[acc]
    save_blacklist("beneficiary", bl)
    return {"message": f"Beneficiary {acc} removed from blacklist"}


# ─── Summary ──────────────────────────────────────────────────────────────────

@router.get("/summary", summary="Blacklist counts")
def blacklist_summary() -> dict:
    return {
        "ip_count":          len(load_blacklist("ip")),
        "device_count":      len(load_blacklist("device")),
        "beneficiary_count": len(load_blacklist("beneficiary")),
    }
