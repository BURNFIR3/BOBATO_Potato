"""
BOB ATO Detection — Hugging Face Space
=======================================
Upload a transaction CSV; the trained behavioral XGBoost pipeline scores
every row and returns ATO risk levels.

Hosted at: https://huggingface.co/spaces/Burnfir3/PotATO
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
import plotly.graph_objects as go
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
        return "SUSPEND", "#f87171"
    elif prob >= 0.50:
        return "OTP REQUIRED", "#facc15"
    else:
        return "ALLOW", "#4ade80"

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
def predict(file_obj):
    empty_df = pd.DataFrame()
    empty_fig = go.Figure()
    
    if file_obj is None:
        return empty_df, "<div style='color:#facc15'>Please upload a CSV file.</div>", empty_fig

    try:
        # Gradio 5.x handles files differently depending on exact versions.
        # It might be a string, a dict, or an object with a .name attribute.
        filepath = getattr(file_obj, "name", file_obj)
        if isinstance(filepath, dict):
            filepath = filepath.get("path", "")
        elif isinstance(filepath, list) and len(filepath) > 0:
            filepath = filepath[0]
            
        df = pd.read_csv(filepath)
    except Exception as e:
        return empty_df, f"<div style='color:#f87171'>Could not read CSV: {e}</div>", empty_fig

    if df.empty:
        return empty_df, "<div style='color:#facc15'>The uploaded file is empty.</div>", empty_fig

    model_type = _detect_model(df)

    try:
        if model_type == "behavioral":
            pipeline = _load_behavioral()
            if pipeline is None:
                return empty_df, "<div style='color:#f87171'>Behavioral model file not found in Space.</div>", empty_fig
            
            # The saved bundle is a dict containing {"pipeline": model, "feature_cols": [...]}
            if isinstance(pipeline, dict) and "pipeline" in pipeline:
                pipeline = pipeline["pipeline"]
                
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
                return empty_df, "<div style='color:#f87171'>Tabular model file not found in Space.</div>", empty_fig
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
            return empty_df, (
                "<div style='color:#f87171'>Could not identify enough matching columns.<br><br>"
                "The model needs at least <strong>10</strong> behavioral feature columns "
                "(e.g. <code>typing_speed_wpm</code>, <code>failed_login_attempts_session</code>) "
                "or <strong>8</strong> tabular feature columns (e.g. <code>velocity_24h</code>, <code>device_fraud_count</code>).<br><br>"
                "Download a sample CSV from the repo to see the expected format.</div>"
            ), empty_fig

    except Exception:
        return empty_df, f"<div style='color:#f87171'>Prediction failed:<br><pre>{traceback.format_exc()}</pre></div>", empty_fig

    # Build results dataframe
    results = df.copy()
    results["ato_probability"] = probs.round(4)
    results["risk_level"] = pd.cut(
        results["ato_probability"],
        bins=[-0.001, 0.499, 0.799, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    results["recommended_action"] = [_action(p)[0] for p in probs]

    # Summary stats
    high = int((results["ato_probability"] >= 0.80).sum())
    med  = int(((results["ato_probability"] >= 0.50) & (results["ato_probability"] < 0.80)).sum())
    low  = int((results["ato_probability"] < 0.50).sum())
    avg  = float(probs.mean())

    summary = f"""
    <div style="margin-bottom:1rem;color:#94a3b8;font-size:0.9rem">Model Used: <strong style="color:#f1f5f9">{model_used}</strong></div>
    <div style="display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap">
      <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.09);border-radius:14px;padding:1.2rem;flex:1;min-width:140px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)">
        <div style="color:#94a3b8;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.5rem">Transactions Scored</div>
        <div style="color:#f1f5f9;font-size:1.8rem;font-weight:700">{len(results):,}</div>
      </div>
      <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.09);border-radius:14px;padding:1.2rem;flex:1;min-width:140px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)">
        <div style="color:#94a3b8;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.5rem">Avg ATO Probability</div>
        <div style="color:#f1f5f9;font-size:1.8rem;font-weight:700">{avg:.4f}</div>
      </div>
    </div>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="background:rgba(248,113,113,.05);border:1px solid rgba(248,113,113,.2);border-radius:14px;padding:1.2rem;flex:1;min-width:120px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)">
        <div style="color:#fca5a5;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.5rem">High Risk (Suspend)</div>
        <div style="color:#f87171;font-size:1.6rem;font-weight:700">{high:,}</div>
      </div>
      <div style="background:rgba(250,204,21,.05);border:1px solid rgba(250,204,21,.2);border-radius:14px;padding:1.2rem;flex:1;min-width:120px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)">
        <div style="color:#fde047;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.5rem">Med Risk (OTP)</div>
        <div style="color:#facc15;font-size:1.6rem;font-weight:700">{med:,}</div>
      </div>
      <div style="background:rgba(74,222,128,.05);border:1px solid rgba(74,222,128,.2);border-radius:14px;padding:1.2rem;flex:1;min-width:120px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)">
        <div style="color:#86efac;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.5rem">Low Risk (Allow)</div>
        <div style="color:#4ade80;font-size:1.6rem;font-weight:700">{low:,}</div>
      </div>
    </div>
    """

    fig = px.histogram(
        results, x="ato_probability", color="recommended_action",
        nbins=40, barmode="overlay",
        color_discrete_map={
            "SUSPEND":      "#f87171",
            "OTP REQUIRED": "#facc15",
            "ALLOW":        "#4ade80",
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
**Links:** [Hugging Face Space](https://huggingface.co/spaces/Burnfir3/PotATO) | [GitHub Repository](https://github.com/BURNFIR3/BOBATO_Potato)

Upload a CSV file containing transaction or session telemetry. The trained XGBoost pipeline will evaluate each record for **Account Takeover risk**.

**Privacy & Security:** No account details or personally identifiable information (PII) are persisted. All data processing occurs strictly in-memory and is immediately discarded post-inference.

### Evaluation Outputs
- `ato_probability` — Model confidence score indicating an ATO attempt (0.0 to 1.0)
- `risk_level` — Categorized risk severity (LOW / MEDIUM / HIGH)
- `recommended_action` — Automated operational directive (ALLOW / OTP REQUIRED / SUSPEND)
"""

ARTICLE = """
### Model Architecture & Methodology

The **behavioral pipeline** evaluates 44 distinct signals, including: 
- Network anomalies (VPN usage, IP distance from historical norms)
- Device signals (unrecognized device fingerprints, historical device fraud count)
- Biometric interaction patterns (typing speed, mouse movement velocity, keystroke timing)
- Session behavior, multi-factor authentication (MFA) failures, and transactional velocity metrics.

The system was trained on 50,000 labeled sessions utilizing SMOTE for class balancing. It achieves the following performance on the held-out test set:
- **AUC-ROC:** 0.953
- **Fraud Precision:** 99.1%
- **Fraud Recall:** 90.1%

*Developed for the Bank of Baroda Fraud Operations Team by Team POTATO.*
"""

with gr.Blocks(
    title="BOB ATO Shield",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#080c14",
        body_text_color="#e2e8f0",
        block_background_fill="#0d1117",
        block_border_color="#1e293b",
    ),
    css="""
    .gr-box { border-radius: 12px !important; }
    .gr-button-primary { background: linear-gradient(135deg, #2563eb, #4f46e5) !important; border: none !important; }
    """,
) as demo:
    gr.HTML("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;padding:1.4rem 1rem 0.9rem;border-bottom:1px solid rgba(255,255,255,0.06)">
      <div style="width:48px;height:48px;background:linear-gradient(135deg,#f97316,#ea580c);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;font-weight:800;color:white;box-shadow:0 4px 14px rgba(249,115,22,.3);flex-shrink:0">B</div>
      <div>
        <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;line-height:1.2">BOB ATO Shield</div>
        <div style="font-size:0.9rem;color:#64748b">Bank of Baroda · Real-time Account Takeover Protection</div>
      </div>
    </div>
    """)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Transaction CSV Data",
                file_types=[".csv"],
            )
            run_btn = gr.Button("Score Transactions", variant="primary", size="lg")

            gr.Markdown("""
            **Supported Feature Schemas:**
            - Behavioral Model (44 columns): e.g., `typing_speed_wpm`, `failed_login_attempts_session`, `device_fraud_count`
            - Tabular Model (31 columns): e.g., `velocity_24h`, `credit_risk_score`, `device_distinct_emails_8w`
            """)

        with gr.Column(scale=2):
            gr.Markdown("### Inference Summary")
            summary_out = gr.HTML()
            chart_out   = gr.Plot(label="Risk Distribution Analysis")

    results_out = gr.Dataframe(
        label="Scored Transaction Results",
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
