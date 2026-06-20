"""
api/main.py  –  FastAPI application entry point.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    OR:
    python scripts/start_api.py
"""

from __future__ import annotations

import sys, os
# Ensure src/ is importable regardless of CWD
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ato_api      import router as ato_router
from blacklist_api import router as blacklist_router
from behavioral_api import router as behavioral_router
from stats_api    import router as stats_router

app = FastAPI(
    title="Bank of Baroda – ATO Detection API",
    description=(
        "Real-time Account Takeover (ATO) detection system. "
        "Uses XGBoost + ATLAS + Behavioral Biometrics. "
        "OTP-only verification (no biometric)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ato_router)
app.include_router(blacklist_router)
app.include_router(behavioral_router)
app.include_router(stats_router)


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "service":  "Bank of Baroda ATO Detection API",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/api/v1/health",
    }
