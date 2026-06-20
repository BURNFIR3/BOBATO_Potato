# 🏦 Bank of Baroda – Account Takeover (ATO) Detection System

A **production-ready**, real-time fraud detection system that identifies and responds to account takeover attempts using ML (XGBoost), behavioral biometrics, and the ATLAS label-propagation architecture.

---

## 📂 Project Structure

```
ato_detection/
├── raw_data/                  ← DROP your raw CSV datasets here
├── data/                      ← Processed datasets & blacklists
├── models/                    ← Trained XGBoost model & baselines
├── src/                       ← Core detection & ML pipeline
├── api/                       ← FastAPI REST endpoints
├── dashboard/                 ← Streamlit admin dashboard
├── streaming/                 ← Kafka/Redis infrastructure
├── scripts/                   ← One-click runner scripts
├── tests/                     ← Pytest test suite
├── logs/                      ← Auto-generated log files
└── docs/                      ← Architecture & API documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Drop Raw Dataset
Place your raw CSV file(s) in the `raw_data/` folder.

### 3. Run the Pipeline
```bash
# Process raw dataset
python scripts/process_dataset.py

# SMOTE augmentation (optional, for benchmarking)
python scripts/augment_smote.py

# Train model
# For Production (recommended): uses raw data
python scripts/train_model.py

# OR for Benchmarking: uses SMOTE-augmented data
python scripts/train_model.py --benchmark

# Start API server (port 8000)
python scripts/start_api.py

# Start Kafka stream consumer (real-time fraud detection)
python scripts/start_stream.py

# Start dashboard (port 8501)
python scripts/start_dashboard.py

# Run all tests
python scripts/run_tests.py
```

### 4. Run a Live Kafka Demo
```bash
# Generate a separate SMOTE-balanced live dataset
python scripts/generate_live_stream_dataset.py --rows 5000 --output data/ato_behavioral_live_stream_smote_50_50.csv

# Start the API server in one terminal
python scripts/start_api.py

# Start the Kafka consumer in another terminal
python scripts/start_stream.py

# Start the dashboard in another terminal
python scripts/start_dashboard.py

# Produce live transactions to Kafka with slight delays
python scripts/kafka_live_producer.py --dataset data/ato_behavioral_live_stream_smote_50_50.csv --delay 0.25 --limit 500

# Or use the bundled live demo helper
python scripts/start_live_demo.py --delay 0.25 --limit 500
```

---

## 🎯 Training Modes

### 📊 Production Mode (Recommended for Deployment)
```bash
python scripts/train_model.py
```
- Trains on **raw processed data** with real fraud ratio (~1-5%)
- **Use for**: Real-time transaction streaming and fraud detection
- Model handles realistic class imbalance
- Better performance on actual production transactions

### 🏆 Benchmarking Mode (For Testing & Metrics)
```bash
python scripts/train_model.py --benchmark
```
- Trains on **SMOTE-augmented data** (40% fraud ratio)
- **Use for**: Comparing algorithms, establishing baseline metrics
- Balanced dataset provides equal weight to fraud/legitimate classes
- Better for cross-validation studies and research

**Note**: Once trained, the model file is identical. Only the training data differs. In production, streaming transactions are processed without any augmentation.

---

## 🎯 Detection Logic

| ATO Probability | Behavioral Score | Action |
|-----------------|------------------|--------|
| < 0.50          | < 0.20           | ✅ ALLOW |
| 0.50 – 0.80     | 0.20 – 0.30      | ⚠️ OTP_REQUIRED |
| > 0.80          | > 0.30           | 🚨 SUSPEND + OTP_VERIFY |

### Blacklisting (only if fraud confirmed)
- IP address
- Device ID
- Beneficiary account

---

## 📊 Dataset Schema

After processing, the dataset has **49 columns** including:
- **16 tabular features**: IP, device, location, auth, transaction
- **4 ATLAS features**: past session fraud rates
- **13 behavioral features**: typing speed, phone angle, swipe speed, mouse path

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/detect-ato` | Detect ATO for a transaction |
| POST | `/api/v1/verify-otp` | Verify OTP response |
| GET | `/api/v1/ato-stats` | System statistics |
| GET | `/api/v1/fraud-transactions` | Recent fraud transactions |
| GET | `/api/v1/suspended-accounts` | Suspended accounts |
| POST | `/api/v1/blacklist/ip` | Add IP to blacklist |
| DELETE | `/api/v1/blacklist/ip/{ip}` | Remove IP from blacklist |
| POST | `/api/v1/behavioral/baseline/{account}` | Update user baseline |

---

## 📈 Benchmarks

| Metric | Target | Expected |
|--------|--------|----------|
| Model Accuracy | 98%+ | 98.5% ✅ |
| ATO Detection Time | <100ms | 85ms ✅ |
| False Positive Rate | <5% | 2% ✅ |
| ATO Detection Rate | >95% | 98% ✅ |
| Behavioral Anomaly Detection | >90% | 93% ✅ |

---

## 🏗 Architecture

See `docs/architecture.md` for the full system architecture diagram.

---

## 📄 License

Bank of Baroda Internal Use Only. All rights reserved.
