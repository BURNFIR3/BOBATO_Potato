"""
src/utils.py
Helper functions: logging, path resolution, blacklist I/O, OTP simulation
"""

import json
import os
import random
import string
from datetime import datetime
from pathlib import Path
from loguru import logger

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # ato_detection/
DATA_DIR        = ROOT / "data"
RAW_DATA_DIR    = ROOT / "raw_data"
MODELS_DIR      = ROOT / "models"
LOGS_DIR        = ROOT / "logs"
BLACKLISTS_DIR  = DATA_DIR / "blacklists"

# ensure directories exist
for _d in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, LOGS_DIR, BLACKLISTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logger(name: str = "ato"):
    log_path = LOGS_DIR / f"{name}.log"
    logger.add(str(log_path), rotation="10 MB", retention="30 days",
               level="INFO", format="{time} | {level} | {message}")
    return logger

# ─── JSON helpers ─────────────────────────────────────────────────────────────
def load_json(filepath: Path) -> dict:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_json(filepath: Path, data: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

# ─── Blacklist helpers ────────────────────────────────────────────────────────
def load_blacklist(name: str) -> dict:
    path = BLACKLISTS_DIR / f"{name}_blacklist.json"
    return load_json(path)

def save_blacklist(name: str, data: dict) -> None:
    path = BLACKLISTS_DIR / f"{name}_blacklist.json"
    save_json(path, data)

def add_to_blacklist(name: str, key: str, meta: dict) -> None:
    bl = load_blacklist(name)
    if key in bl:
        bl[key]["count"] = bl[key].get("count", 0) + 1
        bl[key].update(meta)
    else:
        bl[key] = {"count": 1, **meta}
    bl[key]["last_updated"] = datetime.utcnow().isoformat()
    save_blacklist(name, bl)

def is_blacklisted(name: str, key: str) -> bool:
    bl = load_blacklist(name)
    return key in bl

# ─── OTP simulation ───────────────────────────────────────────────────────────
_otp_store: dict[str, str] = {}   # {account_number: otp}

def generate_otp(account_number: str, length: int = 6) -> str:
    otp = "".join(random.choices(string.digits, k=length))
    _otp_store[account_number] = otp
    logger.info(f"[OTP] Generated OTP for {account_number}: {otp}")
    return otp

def verify_otp(account_number: str, otp: str) -> bool:
    stored = _otp_store.get(account_number)
    if stored and stored == otp:
        del _otp_store[account_number]
        return True
    return False

# ─── Misc ─────────────────────────────────────────────────────────────────────
def generate_transaction_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"TXN-{ts}-{rnd}"

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
