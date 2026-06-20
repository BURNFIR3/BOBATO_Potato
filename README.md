# 🛡️ Bank of Baroda — ATO Detection & Admin Operations Platform

A **production-grade, real-time Account Takeover (ATO) detection system** built for Bank of Baroda fraud operations teams. The platform monitors live transactions, scores them with a trained XGBoost behavioral model, and surfaces decisions to analysts through an admin dashboard — enabling instant intervention on suspicious accounts.

> **Try it online →** Upload a transaction CSV on the [Hugging Face Space](https://huggingface.co/spaces/Burnfir3) to see ATO predictions without any setup.

---

## 📂 Project Structure

```
ato_detection/
├── raw_data/          ← Drop your raw CSV datasets here
├── data/              ← Processed datasets & blacklists
├── models/            ← Trained XGBoost model & pipelines
├── src/               ← Core detection & ML pipeline
├── api/               ← FastAPI REST endpoints (port 8000)
├── dashboard/         ← Streamlit admin dashboard (port 8501)
├── streaming/         ← Kafka/Redis real-time infrastructure
├── scripts/           ← One-click runner scripts
├── tests/             ← Pytest test suite
├── logs/              ← Auto-generated log files
├── docs/              ← Architecture & API documentation
└── hf_space/          ← Hugging Face Space (Gradio demo)
```

---

## 🚀 How to Run — Step by Step

> **Prerequisites:** Python 3.10+, Git. Kafka/Redis are optional (needed only for live streaming).

### Step 1 — Clone & Install

```powershell
git clone https://github.com/BURNFIR3/BOBATO_Potato.git
cd BOBATO_Potato

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate        # Mac/Linux

pip install -r requirements.txt
```

### Step 2 — Add Your Dataset

Place your raw transaction CSV in the `raw_data/` folder:

```
raw_data/
└── your_transactions.csv    ← drop here
```

> If `raw_data/` is empty the system auto-generates 5,000 synthetic records so you can still see the pipeline work.

### Step 3 — Process & Train

```powershell
# Process raw data → data/ato_dataset_processed.csv
python scripts/process_dataset.py

# Train the XGBoost model (~2–5 min)
python scripts/train_model.py
```

### Step 4 — Start the API Server (Terminal 1)

```powershell
python scripts/start_api.py
# → API running at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

### Step 5 — Start the Admin Dashboard (Terminal 2)

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
# → Dashboard at http://localhost:8501
```

### Step 6 — (Optional) Live Streaming with Kafka

Requires Docker for Kafka + Redis:

```powershell
# Terminal 0 — start Kafka + Redis
docker-compose up -d

# Terminal 3 — start Kafka consumer
python scripts/start_stream.py

# Terminal 4 — produce 500 live transactions
python scripts/start_live_demo.py --delay 0.25 --limit 500
```

---

## 🎯 How Decisions Are Made

Every incoming transaction is scored by the behavioral XGBoost pipeline and assigned an ATO probability. The admin dashboard shows the resulting action in real time:

| ATO Probability | Behavioral Signal | Action |
|---|---|---|
| < 0.50 | Low anomaly | ✅ **ALLOW** — transaction proceeds |
| 0.50 – 0.80 | Moderate anomaly | ⚠️ **OTP REQUIRED** — step-up auth triggered |
| > 0.80 | High anomaly | 🚨 **SUSPEND** — account frozen, OTP verify |

When fraud is confirmed, three blacklists are automatically updated:
- **IP address** of the session
- **Device fingerprint** used
- **Beneficiary account** targeted

---

## 📊 Model Performance (Actual Results — June 2026)

### Behavioral Model — Session & Biometric Features
*Trained on 50,000 labeled sessions with 44 behavioral features*

| Metric | Result |
|---|---|
| AUC-ROC | **0.953** |
| Fraud Precision | **99.1%** — when flagged, almost always real fraud |
| Fraud Recall | **90.1%** — catches 9 in 10 actual ATO attempts |
| False Positive Rate | **0.08%** — legitimate users almost never blocked |
| Test Set | 10,000 sessions (20% holdout, unseen during training) |

### Tabular Model — Transaction & Account Features
*Trained on 1,055,000 real bank transactions, 1.5% fraud base rate*

| Metric | Result |
|---|---|
| AUC-ROC | **0.913** |
| Fraud Recall | **73.4%** — catches ~3 in 4 fraud cases |
| False Positive Rate | **9.3%** — conservative, errs on side of caution |
| Test Set | 211,000 transactions (20% holdout) |

> The behavioral model is the **primary scoring engine**. The tabular model is used as a fallback when behavioral signals (mouse/keystroke/touch data) are unavailable.

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/detect-ato` | Score a transaction for ATO risk |
| `POST` | `/api/v1/verify-otp` | Submit OTP verification result |
| `GET` | `/api/v1/ato-stats` | Live operations statistics |
| `GET` | `/api/v1/recent-detections` | Last N scored transactions |
| `GET` | `/api/v1/fraud-transactions` | Confirmed fraud events |
| `GET` | `/api/v1/suspended-accounts` | Currently suspended accounts |
| `POST` | `/api/v1/blacklist/ip` | Add IP to blocklist |
| `DELETE` | `/api/v1/blacklist/ip/{ip}` | Remove IP from blocklist |
| `GET` | `/api/v1/blacklist/summary` | Blacklist counts |

Full interactive docs: **http://localhost:8000/docs**

---

## 🏗 Architecture

```
[Transaction] → [Kafka Topic]
                     ↓
              [Stream Consumer]
                     ↓
          [Behavioral XGBoost Pipeline]
          (44 features: IP, device,
           biometrics, velocity, MFA)
                     ↓
         ┌──────────┬──────────┐
         ↓          ↓          ↓
      ALLOW    OTP_REQUIRED  SUSPEND
                              ↓
                    [Blacklist IP/Device/Beneficiary]
                              ↓
                    [Admin Dashboard — Real-time feed]
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram.

---

## 📁 Dataset Schema

After processing, each record has **44+ columns** including:

- **Network signals**: IP ASN, VPN/proxy flag, IP distance from last login, travel speed
- **Device signals**: new device flag, foreign IP, distinct accounts per device (24h)
- **Behavioral biometrics**: mouse velocity, touch radius, typing speed (WPM), keystroke dwell time
- **Session signals**: page navigation velocity, session length, keep-alive
- **Auth signals**: failed login attempts, password pasted flag, failed MFA count, MFA type changed
- **Profile change signals**: profile details changed, time-to-profile-change
- **Velocity**: tx count (6h / 24h / 4w), failed tx count (1h)
- **Transaction signals**: amount vs historical avg ratio, new payee added, time-to-payout

Full schema: [`docs/dataset_schema.md`](docs/dataset_schema.md)

---

## 🌐 Online Demo (Hugging Face)

No setup required. Upload any CSV with transaction data at:

👉 **https://huggingface.co/spaces/Burnfir3/PotATO** *(deploy from `hf_space/` folder)*

The demo uses the pre-trained behavioral pipeline and returns ATO risk scores for each row.

---

## 📄 License

Bank of Baroda Internal Use Only. All rights reserved.
