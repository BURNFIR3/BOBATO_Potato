"""
BOB ATO Shield — Professional Streamlit dashboard.

Run:
    cd ato_detection
    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

API_BASE = "http://localhost:8000/api/v1"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BOB ATO Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection (split into small safe chunks) ───────────────────────────────
def _css(style: str) -> None:
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

_css(".block-container{padding:1.5rem 2rem 3rem!important;max-width:100%!important}")
_css("[data-testid='stSidebar']{background:linear-gradient(180deg,#0d1117,#0d1520)!important;border-right:1px solid rgba(255,255,255,.06)!important}")
_css("[data-testid='metric-container']{background:rgba(255,255,255,.03)!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:14px!important;padding:1rem 1.2rem!important;transition:border-color .2s,transform .2s}")
_css("[data-testid='metric-container']:hover{border-color:rgba(99,179,237,.35)!important;transform:translateY(-2px)}")
_css("[data-testid='metric-container'] label{color:#94a3b8!important;font-size:.7rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important}")
_css("[data-testid='stMetricValue']{color:#f1f5f9!important;font-size:1.5rem!important;font-weight:700!important}")
_css("[data-testid='stDataFrame']{border-radius:12px!important;border:1px solid rgba(255,255,255,.07)!important}")
_css(".stTabs [data-baseweb='tab']{color:#64748b!important;font-weight:500!important}")
_css(".stTabs [aria-selected='true']{color:#60a5fa!important;border-bottom-color:#3b82f6!important}")
_css("::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.15);border-radius:10px}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def api_get(path: str, default):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

def normalize(records) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=False)
    return df

def dark_fig(fig, **kw):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#94a3b8", size=11),
        margin=dict(l=0, r=0, t=34, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.07)", borderwidth=1),
        **kw,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(0,0,0,0)")
    return fig

ACTION_COLOR = {"ALLOW": "#4ade80", "OTP_REQUIRED": "#facc15", "SUSPEND": "#f87171", "BLOCK": "#f87171"}

def empty_state(icon: str, title: str, sub: str = ""):
    st.markdown(
        f"""<div style="display:flex;flex-direction:column;align-items:center;
        justify-content:center;padding:2.5rem 1rem;text-align:center;
        background:rgba(255,255,255,.015);border:1px dashed rgba(255,255,255,.08);
        border-radius:14px;gap:8px">
        <span style="font-size:1.8rem">{icon}</span>
        <span style="font-size:.88rem;font-weight:600;color:#475569">{title}</span>
        <span style="font-size:.75rem;color:#334155">{sub}</span></div>""",
        unsafe_allow_html=True,
    )

def section_title(label: str):
    st.markdown(
        f"""<div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
        letter-spacing:.08em;color:#475569;margin:0 0 .75rem;display:flex;
        align-items:center;gap:8px">
        <span style="display:inline-block;width:3px;height:13px;
        background:linear-gradient(#3b82f6,#8b5cf6);border-radius:2px"></span>
        {label}</div>""",
        unsafe_allow_html=True,
    )

def stat_row(label: str, value: str):
    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;align-items:center;
        padding:.45rem 0;border-bottom:1px solid rgba(255,255,255,.05)">
        <span style="font-size:.76rem;color:#64748b">{label}</span>
        <span style="font-size:.82rem;font-weight:600;color:#e2e8f0">{value}</span></div>""",
        unsafe_allow_html=True,
    )

def display_full_features(row_data: pd.Series):
    """Displays a clean 3-column view of all features for a reviewed transaction."""
    st.markdown(
        "<div style='padding:1rem; border:1px solid rgba(255,255,255,.08); border-radius:12px; background:rgba(255,255,255,.02)'>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🛡️ Core Identification**")
        core_cols = ["transaction_id", "account_number", "ip_address", "device_id", "ip_country", "ato_probability", "action"]
        for c in core_cols:
            if c in row_data.index:
                stat_row(c, str(row_data[c]))
                
    with col2:
        st.markdown("**🖱️ Behavioral & Network**")
        behav_cols = ["typing_speed_wpm", "mouse_movement_velocity", "touch_event_radius", "is_known_vpn_or_proxy", "is_foreign_ip", "failed_login_attempts_session", "session_length_in_minutes"]
        for c in behav_cols:
            if c in row_data.index:
                stat_row(c, str(row_data[c]))
                
    with col3:
        st.markdown("**💸 Transaction Velocity**")
        txn_cols = ["velocity_6h", "velocity_24h", "velocity_4w", "failed_tx_count_1h", "new_payee_added", "transaction_amount", "transaction_type"]
        for c in txn_cols:
            if c in row_data.index:
                stat_row(c, str(row_data[c]))
                
    # Hidden expander for all remaining raw features
    with st.expander("Show complete raw feature dump"):
        st.json(row_data.to_dict())
    st.markdown("</div>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand block
    st.markdown(
        """<div style="display:flex;align-items:center;gap:12px;
        padding:1.4rem 1rem .9rem;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:1rem">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#f97316,#ea580c);
        border-radius:9px;display:flex;align-items:center;justify-content:center;
        font-size:1rem;font-weight:800;color:white;box-shadow:0 4px 14px rgba(249,115,22,.3);
        flex-shrink:0">B</div>
        <div>
          <div style="font-size:.88rem;font-weight:700;color:#f1f5f9;line-height:1.2">BOB ATO Shield</div>
          <div style="font-size:.68rem;color:#475569">Bank of Baroda · Fraud Ops</div>
        </div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:.09em;color:#334155;padding:0 0 .4rem'>Navigation</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "nav",
        ["⬡  Operations", "◎  Live Detections", "🔎  Transaction Review", "◈  Suspended Accounts", "◉  Blacklists"],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[1]   # strip icon

    st.divider()
    refresh_interval = st.slider("Refresh every (s)", 1, 30, 5)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.markdown(
        f"<div style='font-size:.67rem;color:#1e293b;padding-top:.5rem'>"
        f"↻ {datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )

# ── Fetch data ─────────────────────────────────────────────────────────────────
stats     = api_get("/ato-stats", {})
recent    = normalize(api_get("/recent-detections", []))
fraud     = normalize(api_get("/fraud-transactions", []))
suspended = normalize(api_get("/suspended-accounts", []))

is_online = stats.get("status", "Offline") == "Online"
dot_html  = (
    '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
    'background:#4ade80;box-shadow:0 0 6px #4ade80;margin-right:6px;'
    'animation:none"></span>Online'
    if is_online else
    '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
    'background:#f87171;box-shadow:0 0 6px #f87171;margin-right:6px"></span>Offline'
)

# ── Top header ─────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown("## 🛡️ ATO Detection Dashboard")
    st.caption("Real-time account takeover protection · Bank of Baroda")
with hcol2:
    badge_bg = "rgba(34,197,94,.1)" if is_online else "rgba(239,68,68,.1)"
    badge_bd = "rgba(34,197,94,.3)" if is_online else "rgba(239,68,68,.3)"
    badge_col = "#4ade80" if is_online else "#f87171"
    st.markdown(
        f"""<div style="margin-top:.9rem;display:flex;justify-content:flex-end">
        <span style="display:inline-flex;align-items:center;padding:5px 14px;
        border-radius:999px;background:{badge_bg};border:1px solid {badge_bd};
        font-size:.72rem;font-weight:600;color:{badge_col}">{dot_html}</span></div>""",
        unsafe_allow_html=True,
    )
st.markdown("<hr style='border-color:rgba(255,255,255,.06);margin:0 0 1.5rem'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Operations":
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    action_counts = stats.get("action_counts", {})
    total_decisions = sum(action_counts.values()) if action_counts else 0
    blocked = action_counts.get("SUSPEND", 0) + action_counts.get("BLOCK", 0)
    otp_count = action_counts.get("OTP_REQUIRED", 0)
    flagged_for_review = blocked + otp_count
    
    c1.metric("📋 Decisions Today",  total_decisions)
    c2.metric("🚩 Flagged Txns",     stats.get("fraud_transactions_count", 0))
    c3.metric("⚠️ Flagged for Review", flagged_for_review)
    c4.metric("🚨 Blocked",          blocked)
    c5.metric("📲 OTP Triggered",    otp_count)
    c6.metric("🤖 Engine",           "✅ Online" if stats.get("live_behavioral_model_loaded") else "⚠️ Offline")

    st.markdown("<div style='margin-top:1.3rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="medium")

    with left:
        section_title("Regional Demographics (Flagged Traffic)")
        if not fraud.empty and "ip_country" in fraud.columns:
            # Generate a chloropleth map for regions
            country_counts = fraud["ip_country"].value_counts().reset_index()
            country_counts.columns = ["Country", "Count"]
            
            fig_map = px.choropleth(
                country_counts,
                locations="Country",
                locationmode="country names",
                color="Count",
                color_continuous_scale="Oranges",
                title="Geographic Fraud Distribution"
            )
            # Custom dark styling for the map
            fig_map.update_geos(
                bgcolor="rgba(0,0,0,0)",
                showcoastlines=True, coastlinecolor="rgba(255,255,255,0.1)",
                showland=True, landcolor="rgba(255,255,255,0.02)",
                showcountries=True, countrycolor="rgba(255,255,255,0.1)"
            )
            dark_fig(fig_map)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            empty_state("🌍", "No regional data available", "Map will populate when flagged transactions with IP country data arrive.")

    with right:
        section_title("Action Mix")
        if action_counts:
            df_ac = pd.DataFrame([{"Action": k, "Count": v} for k, v in action_counts.items()])
            color_map = {k: ACTION_COLOR.get(k, "#60a5fa") for k in df_ac["Action"]}
            fig = px.pie(df_ac, values="Count", names="Action", hole=0.58,
                         color="Action", color_discrete_map=color_map)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              marker=dict(line=dict(color="#080c14", width=2)))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("📊", "Waiting for traffic", "Action distribution appears after live data arrives")


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DETECTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Detections":
    if not recent.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Events", len(recent))
        if "ato_probability" in recent.columns:
            k2.metric("Avg Risk Score", f"{recent['ato_probability'].mean():.3f}")
            k3.metric("High-Risk (>80%)", int((recent["ato_probability"] > 0.8).sum()))
        st.markdown("<div style='margin-top:.8rem'></div>", unsafe_allow_html=True)

    t1, t2 = st.tabs(["🔴  Live Feed", "🚩  Flagged Transactions"])

    with t1:
        section_title("Live Detection Feed")
        if recent.empty:
            empty_state("📡", "No live detections", "Waiting for Kafka stream…")
        else:
            cols = ["timestamp", "transaction_id", "account_number", "ip_country", "action", "ato_probability"]
            st.dataframe(recent[[c for c in cols if c in recent.columns]], use_container_width=True, height=300, hide_index=True)
            if {"ato_probability", "action"}.issubset(recent.columns):
                fig = px.histogram(
                    recent, x="ato_probability", color="action", nbins=30,
                    barmode="overlay", color_discrete_map=ACTION_COLOR,
                    title="ATO Risk Score Distribution",
                )
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

    with t2:
        section_title("Flagged Transactions")
        if fraud.empty:
            empty_state("✅", "No flagged transactions", "All clear — no fraud recorded")
        else:
            cols = ["timestamp", "transaction_id", "account_number", "ip_country", "action", "ato_probability"]
            st.dataframe(fraud[[c for c in cols if c in fraud.columns]], use_container_width=True, height=350, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTION REVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transaction Review":
    st.markdown("### 🔎 Analyst Review Console")
    st.caption("Deep dive into flagged transactions and review raw behavioral signals.")
    
    if fraud.empty:
        empty_state("✅", "No flagged transactions", "No transactions currently require review.")
    else:
        # Create a dropdown for selection
        txn_ids = fraud["transaction_id"].tolist()
        
        col_sel, col_empty = st.columns([2, 2])
        with col_sel:
            selected_txn = st.selectbox("Select Flagged Transaction ID to Review:", txn_ids)
            
        if selected_txn:
            row_data = fraud[fraud["transaction_id"] == selected_txn].iloc[0]
            
            st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
            section_title(f"Reviewing Transaction: {selected_txn}")
            
            # Display action pill
            action = row_data.get("action", "UNKNOWN")
            color = ACTION_COLOR.get(action, "#64748b")
            st.markdown(
                f"""<div style="display:inline-block; padding:4px 12px; background:rgba(255,255,255,0.05); 
                border:1px solid {color}; border-radius:12px; color:{color}; font-weight:700; font-size:0.85rem; margin-bottom:1rem;">
                Recommended Action: {action}
                </div>""", 
                unsafe_allow_html=True
            )
            
            # Show the detailed feature breakdown
            display_full_features(row_data)


# ══════════════════════════════════════════════════════════════════════════════
#  SUSPENDED ACCOUNTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Suspended Accounts":
    n = len(suspended) if not suspended.empty else 0
    badge_col = "#f87171" if n > 0 else "#4ade80"
    badge_bg  = "rgba(239,68,68,.1)" if n > 0 else "rgba(34,197,94,.1)"
    badge_bd  = "rgba(239,68,68,.3)" if n > 0 else "rgba(34,197,94,.3)"
    hdl, hdr = st.columns([5, 1])
    with hdl:
        st.markdown("### 🔒 Suspended Accounts")
        st.caption("Accounts with confirmed or high-probability ATO activity")
    with hdr:
        st.markdown(
            f"""<div style="margin-top:1rem;display:flex;justify-content:flex-end">
            <span style="display:inline-flex;align-items:center;padding:5px 14px;
            border-radius:999px;background:{badge_bg};border:1px solid {badge_bd};
            font-size:.72rem;font-weight:600;color:{badge_col}">{n} Account{'s' if n!=1 else ''}</span></div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border-color:rgba(255,255,255,.06);margin:.4rem 0 1rem'>", unsafe_allow_html=True)
    if suspended.empty:
        empty_state("✅", "No suspended accounts", "All accounts are currently in good standing")
    else:
        st.dataframe(suspended, use_container_width=True, height=460, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  BLACKLISTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Blacklists":
    summary = api_get("/blacklist/summary", {})
    b1, b2, b3 = st.columns(3)
    b1.metric("🌐 Blacklisted IPs",        summary.get("ip_count", 0))
    b2.metric("📱 Blacklisted Devices",    summary.get("device_count", 0))
    b3.metric("👤 Blacklisted Beneficiaries", summary.get("beneficiary_count", 0))

    st.markdown("<div style='margin-top:.9rem'></div>", unsafe_allow_html=True)
    tb1, tb2, tb3 = st.tabs(["🌐  IP Addresses", "📱  Devices", "👤  Beneficiaries"])
    endpoints = ["/blacklist/ip", "/blacklist/device", "/blacklist/beneficiary"]
    labels    = ["ip_address",   "device_id",         "beneficiary_account"]
    for tab, ep, lbl in zip([tb1, tb2, tb3], endpoints, labels):
        with tab:
            data = api_get(ep, {})
            if not data:
                empty_state("🧹", "Blacklist is empty", "Entries appear here after confirmed fraud")
            else:
                rows = [{lbl: k, **v} for k, v in data.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Auto refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
