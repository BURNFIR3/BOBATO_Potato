"""
BOB ATO Detection — Hugging Face Space
=======================================
Upload a transaction CSV; the trained behavioral XGBoost pipeline scores
every row and returns ATO risk levels.

Hosted at: https://huggingface.co/spaces/Burnfir3/bob-ato-detection
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

# Behavioral pipeline (primary — 44 features)
BEHAVIORAL_PIPELINE_PATH = BASE / "models" / "ato_behavioral_live_pipeline.joblib"
BEHAVIORAL_METADATA_PATH = BASE / "models" / "ato_behavioral_live_metadata.json"

# Tabular XGBoost model (fallback — 31 raw features)
TABULAR_MODEL_PATH = BASE / "models" / "ato_xgboost_model.json"

# ── Feature sets ──────────────────────────────────────────────────────────────
BEHAVIORAL_FEATURES = [
    "ip_address_asn", "is_known_vpn_or_proxy", "ip_distance_from_last_login_km",
    "speed_of_travel_kmh", "is_new_device_uid", "is_foreign_ip",
    "country_mismatch_with_profile", "distinct_ip_zip_count_1w_4w",
    "login_hour_of_day_local", "login_hour_anomaly_score",
    "mouse_movement_velocity", "touch_event_radius", "typing_speed_wpm",
    "keystroke_dwell_time", "failed_login_attempts_session", "password_pasted",
    "page_navigation_velocity", "session_length_in_minutes", "keep_alive_session",
    "device_distinct_emails_8w", "device_distinct_accounts_attempted_24h",
    "device_fraud_count", "customer_age", "account_age_months",
    "days_since_last_login", "days_since_password_reset", "email_is_free",
    "name_email_similarity", "phone_home_valid", "phone_mobile_valid",
    "failed_mfa_attempts_count", "mfa_type_changed", "profile_details_changed_flag",
    "time_to_profile_change_seconds", "velocity_6h", "velocity_24h", "velocity_4w",
    "failed_tx_count_1h", "tx_amount_vs_historical_avg_ratio", "new_payee_added",
    "time_from_payee_addition_to_payout_minutes",
    "sequential_failed_promocode_attempts", "saved_payment_methods_count", "month",
]

TABULAR_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
    "customer_age", "days_since_request", "intended_balcon_amount", "payment_type",
    "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w", "new_browser",
    "foreign_request", "amount", "keep_alive_session", "proposed_credit_limit",
    "employment_status", "credit_risk_score", "email_is_free", "housing_status",
    "phone_home_valid", "phone_mobile_valid", "has_other_cards", "source",
    "device_os", "device_distinct_emails_8w", "month", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "device_fraud_count",
]

# ── Load models ────────────────────────────────────────────────────────────────
_beh_pipeline = None
_tab_model = None

def _load_behavioral():
    global _beh_pipeline
    if _beh_pipeline is None and BEHAVIORAL_PIPELINE_PATH.exists():
        _beh_pipeline = joblib.load(BEHAVIORAL_PIPELINE_PATH)
    return _beh_pipeline

def _load_tabular():
    global _tab_model
    if _tab_model is None and TABULAR_MODEL_PATH.exists():
        _tab_model = xgb.XGBClassifier()
        _tab_model.load_model(str(TABULAR_MODEL_PATH))
    return _tab_model

# ── Decision logic ─────────────────────────────────────────────────────────────
def _action(prob: float) -> tuple[str, str]:
    if prob >= 0.80:
        return "🚨 SUSPEND", "#f87171"
    elif prob >= 0.50:
        return "⚠️ OTP REQUIRED", "#facc15"
    else:
        return "✅ ALLOW", "#4ade80"

def _detect_model(df: pd.DataFrame) -> str:
    """Return 'behavioral', 'tabular', or 'unknown'."""
    beh_cols = [c for c in BEHAVIORAL_FEATURES if c in df.columns]
    tab_cols  = [c for c in TABULAR_FEATURES  if c in df.columns]
    if len(beh_cols) >= 10:
        return "behavioral"
    if len(tab_cols) >= 8:
        return "tabular"
    return "unknown"

# ── Core prediction ────────────────────────────────────────────────────────────
def predict(file_obj) -> tuple:
    if file_obj is None:
        return None, "⚠️ Please upload a CSV file.", None

    try:
        df = pd.read_csv(file_obj.name)
    except Exception as e:
        return None, f"❌ Could not read CSV: {e}", None

    if df.empty:
        return None, "❌ The uploaded file is empty.", None

    model_type = _detect_model(df)

    try:
        if model_type == "behavioral":
            pipeline = _load_behavioral()
            if pipeline is None:
                return None, "❌ Behavioral model file not found in Space.", None
            feature_cols = [c for c in BEHAVIORAL_FEATURES if c in df.columns]
            X = df[feature_cols].copy()
            # Fill missing features with 0
            for col in BEHAVIORAL_FEATURES:
                if col not in X.columns:
                    X[col] = 0
            X = X[BEHAVIORAL_FEATURES]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
            probs = pipeline.predict_proba(X)[:, 1]
            model_used = "Behavioral Pipeline (44 features)"

        elif model_type == "tabular":
            model = _load_tabular()
            if model is None:
                return None, "❌ Tabular model file not found in Space.", None
            feature_cols = [c for c in TABULAR_FEATURES if c in df.columns]
            X = df[feature_cols].copy()
            for col in TABULAR_FEATURES:
                if col not in X.columns:
                    X[col] = 0
            X = X[TABULAR_FEATURES]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
            probs = model.predict_proba(X)[:, 1]
            model_used = "Tabular XGBoost (31 features)"

        else:
            return None, (
                "❌ Could not identify enough matching columns.\n\n"
                "The model needs at least **10** behavioral feature columns "
                "(e.g. `typing_speed_wpm`, `failed_login_attempts_session`) "
                "or **8** tabular feature columns (e.g. `velocity_24h`, `device_fraud_count`).\n\n"
                "Download a sample CSV from the repo to see the expected format."
            ), None

    except Exception:
        return None, f"❌ Prediction failed:\n```\n{traceback.format_exc()}\n```", None

    # Build results dataframe
    results = df.copy()
    results["ato_probability"] = probs.round(4)
    results["risk_level"] = pd.cut(
        results["ato_probability"],
        bins=[-0.001, 0.499, 0.799, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    results["recommended_action"] = [_action(p)[0] for p in probs]

    # Summary stats
    high = int((results["ato_probability"] >= 0.80).sum())
    med  = int(((results["ato_probability"] >= 0.50) & (results["ato_probability"] < 0.80)).sum())
    low  = int((results["ato_probability"] < 0.50).sum())
    avg  = float(probs.mean())

    summary = (
        f"### ✅ Scored {len(results):,} transactions  ·  Model: *{model_used}*\n\n"
        f"| Risk Level | Count | Action |\n"
        f"|---|---|---|\n"
        f"| 🚨 HIGH (≥ 0.80) | **{high}** | SUSPEND |\n"
        f"| ⚠️ MEDIUM (0.50–0.80) | **{med}** | OTP REQUIRED |\n"
        f"| ✅ LOW (< 0.50) | **{low}** | ALLOW |\n\n"
        f"**Average ATO probability:** `{avg:.4f}`"
    )

    # Risk distribution chart
    fig = px.histogram(
        results, x="ato_probability", color="recommended_action",
        nbins=40, barmode="overlay",
        color_discrete_map={
            "🚨 SUSPEND":      "#f87171",
            "⚠️ OTP REQUIRED": "#facc15",
            "✅ ALLOW":        "#4ade80",
        },
        title="ATO Risk Score Distribution",
        labels={"ato_probability": "ATO Probability", "count": "Transactions"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend_title_text="Action",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Output table — put key columns first
    key_cols = ["ato_probability", "risk_level", "recommended_action"]
    id_cols  = [c for c in ["account_number", "transaction_id", "customer_id"] if c in results.columns]
    other    = [c for c in results.columns if c not in key_cols + id_cols]
    out = results[id_cols + key_cols + other]

    return out, summary, fig


# ── Gradio UI ──────────────────────────────────────────────────────────────────
DESCRIPTION = """
## 🛡️ Bank of Baroda — ATO Detection

Upload a CSV file of transactions or sessions and the trained XGBoost model will score each row for **Account Takeover risk**.

**No account details or PII are stored.** All processing happens in-memory and is discarded after each request.

### What you get back
- `ato_probability` — model confidence that this is an ATO attempt (0–1)
- `risk_level` — LOW / MEDIUM / HIGH
- `recommended_action` — ALLOW / OTP REQUIRED / SUSPEND
"""

ARTICLE = """
### How the model works

The **behavioral pipeline** looks at 44 signals: network anomalies (VPN use, IP distance),
device signals (new device, device fraud history), biometric patterns (typing speed, mouse velocity,
keystroke timing), session behaviour, MFA failures, velocity metrics, and transaction anomalies.

Trained on 50,000 labeled sessions with SMOTE balancing. Achieves **AUC-ROC 0.953**,
**99.1% fraud precision**, and **90.1% fraud recall** on the held-out test set.

> Built for the Bank of Baroda Fraud Operations team · [GitHub](https://github.com/BURNFIR3/BOBATO_Potato)
"""

with gr.Blocks(
    title="BOB ATO Detection",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#080c14",
        body_text_color="#e2e8f0",
        block_background_fill="#0d1117",
        block_border_color="#1e293b",
    ),
    css="""
    .gr-box { border-radius: 12px !important; }
    .gr-button-primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; border: none !important; }
    """,
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📂 Upload Transaction CSV",
                file_types=[".csv"],
                type="filepath",
            )
            run_btn = gr.Button("🔍 Score Transactions", variant="primary", size="lg")

            gr.Markdown("""
            **Supported feature sets:**
            - Behavioral (44 cols): `typing_speed_wpm`, `failed_login_attempts_session`, `device_fraud_count`, …
            - Tabular (31 cols): `velocity_24h`, `credit_risk_score`, `device_distinct_emails_8w`, …
            """)

        with gr.Column(scale=2):
            summary_out = gr.Markdown(label="Summary")
            chart_out   = gr.Plot(label="Risk Distribution")

    results_out = gr.Dataframe(
        label="Scored Transactions",
        interactive=False,
        wrap=False,
    )

    run_btn.click(
        fn=predict,
        inputs=[file_input],
        outputs=[results_out, summary_out, chart_out],
    )

    gr.Markdown(ARTICLE)

if __name__ == "__main__":
    demo.launch()
