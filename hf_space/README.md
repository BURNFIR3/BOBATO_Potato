---
title: BOB ATO Detection
emoji: 🛡️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Upload a transaction CSV to get real-time ATO risk scores
---

# 🛡️ Bank of Baroda — ATO Detection Demo

Upload a CSV file of bank transactions and the trained XGBoost behavioral model will score each row for **Account Takeover (ATO) risk**, returning a recommended action for each transaction.

## How it works

1. Upload a CSV with transaction/session features
2. The model auto-detects available features and selects the best pipeline
3. Each row is scored and assigned a risk level: `ALLOW`, `OTP_REQUIRED`, or `SUSPEND`

## Sample columns the model uses

`ip_address_asn`, `is_known_vpn_or_proxy`, `typing_speed_wpm`, `failed_login_attempts_session`, `device_fraud_count`, `velocity_24h`, `new_payee_added`, and 37 more behavioral signals.

Download a sample CSV from the repo to try it out.
