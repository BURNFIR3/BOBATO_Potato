# 🚀 Deployment Guide – Bank of Baroda ATO Detection

## Prerequisites

- Python 3.10+
- (Optional) Apache Kafka 3.x
- (Optional) Redis 7.x

---

## Step-by-Step Setup

### 1. Install Python Dependencies

```bash
cd ato_detection
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` to set your Redis/Kafka hostnames if different from localhost.

### 3. Drop Raw Dataset

Place your raw CSV files in:
```
ato_detection/raw_data/
```

### 4. Run the Full Pipeline

```bash
# Process raw dataset (auto-generates synthetic data if raw_data/ is empty)
python scripts/process_dataset.py

# SMOTE augmentation (target: 40% fraud ratio)
python scripts/augment_smote.py

# Train XGBoost model (~2-5 minutes)
python scripts/train_model.py

# Start API server (port 8000)
python scripts/start_api.py

# (Optional) Start streaming consumer
python scripts/start_stream.py

# Start admin dashboard (port 8501)
python scripts/start_dashboard.py
```

### 5. Access the Services

| Service | URL |
|---------|-----|
| API Docs | http://localhost:8000/docs |
| Admin Dashboard | http://localhost:8501 |
| Health Check | http://localhost:8000/api/v1/health |

---

## Docker Deployment (Optional)

```yaml
# docker-compose.yml (minimal)
version: '3.8'
services:
  ato-api:
    build: .
    command: python scripts/start_api.py
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs

  ato-dashboard:
    build: .
    command: python scripts/start_dashboard.py
    ports:
      - "8501:8501"
    depends_on:
      - ato-api
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model not found` | Run `python scripts/train_model.py` first |
| `No CSV in raw_data/` | System auto-generates 5,000 synthetic records |
| `kafka-python error` | Install: `pip install kafka-python` |
| `redis error` | Install: `pip install redis` |
| `imbalanced-learn error` | Install: `pip install imbalanced-learn` |
