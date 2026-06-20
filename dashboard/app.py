"""
Professional Streamlit dashboard for live ATO monitoring.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="BOB ATO Operations",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, Segoe UI, Arial, sans-serif;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #f7f8fa;
    border-right: 1px solid #e5e7eb;
}
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.5rem !important;
}
.header {
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.8rem;
    margin-bottom: 1rem;
}
.header h1 {
    font-size: 1.55rem;
    margin: 0;
    color: #0f172a;
}
.header p {
    margin: 0.25rem 0 0 0;
    color: #64748b;
}
.status-ok {
    color: #166534;
    font-weight: 700;
}
.status-bad {
    color: #991b1b;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


def api_get(path: str, default):
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception:
        return default


def normalize_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=False)
    return df


with st.sidebar:
    st.markdown("### BOB ATO Operations")
    refresh_interval = st.slider("Refresh interval", 1, 15, 3)
    auto_refresh = st.checkbox("Auto refresh", value=True)
    st.divider()
    page = st.radio(
        "View",
        [
            "Operations",
            "Live Detections",
            "Suspended Accounts",
            "Blacklists",
            "Model Metrics",
        ],
    )
    st.divider()
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


stats = api_get("/ato-stats", {})
status = stats.get("status", "Offline")
status_class = "status-ok" if status == "Online" else "status-bad"

st.markdown(
    f"""
<div class="header">
  <h1>Bank of Baroda ATO Detection Dashboard</h1>
  <p>Live detection service status: <span class="{status_class}">{status}</span></p>
</div>
""",
    unsafe_allow_html=True,
)

recent = normalize_records(api_get("/recent-detections", []))
fraud = normalize_records(api_get("/fraud-transactions", []))
suspended = normalize_records(api_get("/suspended-accounts", []))

if page == "Operations":
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Recent Detections", stats.get("recent_detections_count", 0))
    c2.metric("Flagged Transactions", stats.get("fraud_transactions_count", 0))
    c3.metric("Suspended Accounts", stats.get("suspended_accounts_count", 0))
    c4.metric("Model AUC", stats.get("model_auc", "0.0000"))
    c5.metric("Model F1", stats.get("model_f1", "0.0000"))
    c6.metric("Live Model", "Loaded" if stats.get(
        "live_behavioral_model_loaded") else "Missing")

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Most Recent Decisions")
        if recent.empty:
            st.info("No live detections yet. Start the Kafka producer and consumer.")
        else:
            cols = [
                "timestamp",
                "transaction_id",
                "account_number",
                "action",
                "ato_probability",
                "actual_is_ato",
            ]
            st.dataframe(recent[[c for c in cols if c in recent.columns]].head(
                25), use_container_width=True)

    with right:
        st.subheader("Action Mix")
        action_counts = stats.get("action_counts", {})
        if action_counts:
            action_df = pd.DataFrame(
                [{"action": k, "count": v} for k, v in action_counts.items()]
            )
            fig = px.bar(action_df, x="action", y="count", color="action")
            fig.update_layout(showlegend=False,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Action distribution will appear after live detections arrive.")

elif page == "Live Detections":
    st.subheader("Live Detection Feed")
    if recent.empty:
        st.info("No live detections available.")
    else:
        st.dataframe(recent, use_container_width=True)
        if {"ato_probability", "action"}.issubset(recent.columns):
            fig = px.histogram(
                recent,
                x="ato_probability",
                color="action",
                nbins=20,
                title="ATO Probability Distribution",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged Transactions")
    if fraud.empty:
        st.info("No flagged transactions have been recorded.")
    else:
        st.dataframe(fraud, use_container_width=True)

elif page == "Suspended Accounts":
    st.subheader("Suspended Accounts")
    if suspended.empty:
        st.success("No accounts are currently suspended.")
    else:
        st.dataframe(suspended, use_container_width=True)

elif page == "Blacklists":
    st.subheader("Blacklist Summary")
    summary = api_get("/blacklist/summary", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("IP Addresses", summary.get("ip_count", 0))
    c2.metric("Devices", summary.get("device_count", 0))
    c3.metric("Beneficiaries", summary.get("beneficiary_count", 0))

    tabs = st.tabs(["IP Addresses", "Devices", "Beneficiaries"])
    endpoints = ["/blacklist/ip",
                 "/blacklist/device", "/blacklist/beneficiary"]
    labels = ["ip_address", "device_id", "beneficiary_account"]
    for tab, endpoint, label in zip(tabs, endpoints, labels):
        with tab:
            data = api_get(endpoint, {})
            if not data:
                st.info("No records.")
            else:
                rows = [{label: key, **value} for key, value in data.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif page == "Model Metrics":
    st.subheader("Model Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", stats.get("model_accuracy", "0.0%"))
    c2.metric("AUC-ROC", stats.get("model_auc", "0.0000"))
    c3.metric("F1 Score", stats.get("model_f1", "0.0000"))
    c4.metric("False Positive Rate", stats.get("false_positive_rate", "0.00%"))

    st.markdown(
        """
The live service uses the saved behavioral pipeline when incoming Kafka
transactions contain behavioral dataset fields. The bundle includes median
imputation, scaling, and the trained XGBoost classifier, so dashboard traffic is
scored with the same preprocessing used during benchmark training.
"""
    )

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
