# Bank of Baroda — Account Takeover (ATO) Detection & Admin Operations Platform

A production-grade, real-time Account Takeover (ATO) detection system developed for Bank of Baroda fraud operations teams. The platform monitors transactions, scores them using a trained XGBoost behavioral model, and surfaces decisions to analysts through an administrative dashboard, enabling immediate intervention on suspicious accounts.

> **Online Deployment:** Access the live prediction environment via [Hugging Face Space](https://huggingface.co/spaces/Burnfir3/PotATO).

---

## Project Structure

```
ato_detection/
├── raw_data/          ← Input directory for raw CSV datasets
├── data/              ← Processed datasets and blacklists
├── models/            ← Trained XGBoost models and pipelines
├── src/               ← Core detection and ML pipeline logic
├── api/               ← FastAPI REST endpoints (port 8000)
├── dashboard/         ← Streamlit administrative dashboard (port 8501)
├── streaming/         ← Kafka/Redis real-time infrastructure
├── scripts/           ← Execution and operational scripts
├── tests/             ← Pytest verification suite
├── logs/              ← Automated system logs
├── docs/              ← Architecture and API documentation
└── hf_space/          ← Hugging Face Space deployment repository
```

---

## Operational Instructions

> **Prerequisites:** Python 3.10+, Git. Kafka/Redis are required only for live streaming components.

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

### Step 2 — Dataset Ingestion

Place the raw transaction CSV in the `raw_data/` directory:

```
raw_data/
└── your_transactions.csv
```

> If `raw_data/` is empty, the system automatically generates synthetic records for demonstration purposes.

### Step 3 — Process & Train

```powershell
# Process raw data → data/ato_dataset_processed.csv
python scripts/process_dataset.py

# Train the XGBoost model
python scripts/train_model.py
```

### Step 4 — Start the API Server (Terminal 1)

```powershell
python scripts/start_api.py
# → API running at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

### Step 5 — Start the Administrative Dashboard (Terminal 2)

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
# → Dashboard at http://localhost:8501
```

### Step 6 — (Optional) Live Streaming with Kafka

Requires Docker for Kafka and Redis configurations:

```powershell
# Terminal 0 — Initialize Kafka and Redis
docker-compose up -d

# Terminal 3 — Initialize Kafka consumer
python scripts/start_stream.py

# Terminal 4 — Produce live transactions
python scripts/start_live_demo.py --delay 0.25 --limit 500
```

---

## Decision Logic

Incoming transactions are scored by the behavioral XGBoost pipeline and assigned an ATO probability. The administrative dashboard reflects the resulting action in real time:

| ATO Probability | Behavioral Signal | Action |
|---|---|---|
| < 0.50 | Low anomaly | **ALLOW** — transaction proceeds |
| 0.50 – 0.80 | Moderate anomaly | **OTP REQUIRED** — step-up authentication triggered |
| > 0.80 | High anomaly | **SUSPEND** — account frozen, pending verification |

Confirmed fraud automatically updates three primary blacklists:
- **IP address** of the associated session
- **Device fingerprint** identifier
- **Beneficiary account** targeted

---

## Model Performance

### Behavioral Model — Session & Biometric Features
*Trained on 50,000 labeled sessions leveraging 44 behavioral features*

| Metric | Result |
|---|---|
| AUC-ROC | **0.953** |
| Fraud Precision | **99.1%** |
| Fraud Recall | **90.1%** |
| False Positive Rate | **0.08%** |
| Test Set | 10,000 sessions (20% holdout distribution) |

### Tabular Model — Transaction & Account Features
*Trained on 1,055,000 transactions utilizing a 1.5% fraud base rate*

| Metric | Result |
|---|---|
| AUC-ROC | **0.913** |
| Fraud Recall | **73.4%** |
| False Positive Rate | **9.3%** |
| Test Set | 211,000 transactions (20% holdout distribution) |

> The behavioral model operates as the primary scoring engine. The tabular model functions as a strict fallback mechanism when behavioral signals (mouse/keystroke/touch data) are unavailable.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/detect-ato` | Evaluate transaction for ATO risk |
| `POST` | `/api/v1/verify-otp` | Submit OTP verification result |
| `GET` | `/api/v1/ato-stats` | Retrieve operations statistics |
| `GET` | `/api/v1/recent-detections` | Retrieve recent transaction scores |
| `GET` | `/api/v1/fraud-transactions` | Retrieve confirmed fraud events |
| `GET` | `/api/v1/suspended-accounts` | Retrieve suspended accounts list |
| `POST` | `/api/v1/blacklist/ip` | Add IP address to blocklist |
| `DELETE` | `/api/v1/blacklist/ip/{ip}` | Remove IP address from blocklist |
| `GET` | `/api/v1/blacklist/summary` | Retrieve blacklist metric counts |

Full interactive documentation is available at: **http://localhost:8000/docs**

---

## Architecture

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
                    [Administrative Dashboard]
```

Please refer to [`docs/architecture.md`](docs/architecture.md) for comprehensive architectural details.

---

## Dataset Schema

Processed records utilize a schema of 44+ columns, including:

- **Network signals**: IP ASN, VPN/proxy flags, IP distance, travel speed
- **Device signals**: device novelty, foreign IP presence, distinct accounts per device
- **Behavioral biometrics**: mouse velocity, touch radius, typing speed (WPM), keystroke dwell time
- **Session signals**: navigation velocity, session duration, keep-alive monitoring
- **Authentication signals**: failed login attempts, pasted password flags, failed MFA counts, MFA configuration changes
- **Profile signals**: profile alterations, time-to-profile-change
- **Velocity metrics**: transaction frequencies across 6h, 24h, and 4w intervals
- **Transaction signals**: transaction value ratios, new payee additions, time-to-payout execution

The complete schema documentation is available at: [`docs/dataset_schema.md`](docs/dataset_schema.md)

---

## Online Environment Deployment

**Access URL:** https://huggingface.co/spaces/Burnfir3/PotATO

The deployment utilizes the pre-trained behavioral pipeline to return ATO risk severity evaluations.

---

## Note

Some features of the dataset have been synthetically generated to include more behavioural features which are not freely available on the internet. The script can be found at ato_detection\src\behavioral_dataset_generator.py

---

## License

Bank of Baroda Internal Use Only. All rights reserved.
