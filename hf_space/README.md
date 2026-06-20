---
title: BOB ATO Detection
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Account Takeover (ATO) Detection System
---

# Bank of Baroda — Account Takeover (ATO) Detection System

Upload a CSV file containing transaction data. The trained XGBoost behavioral pipeline evaluates each record for **Account Takeover (ATO) risk**, determining the appropriate automated response for the transaction.

## System Workflow

1. Data Ingestion: Upload a CSV containing transaction or session features.
2. Automated Feature Extraction: The system detects available features and selects the optimal processing pipeline.
3. Risk Assessment: Each record is scored and assigned a categorized action: `ALLOW`, `OTP_REQUIRED`, or `SUSPEND`.

## Processed Telemetry Signals

The model leverages over 44 behavioral signals, including:
`ip_address_asn`, `is_known_vpn_or_proxy`, `typing_speed_wpm`, `failed_login_attempts_session`, `device_fraud_count`, `velocity_24h`, and `new_payee_added`.
