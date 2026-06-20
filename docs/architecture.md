# 🏗 System Architecture – Bank of Baroda ATO Detection

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRANSACTION ENTRY                          │
│   Mobile App / Web Portal / ATM / Branch                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                             │
│   POST /api/v1/detect-ato                                           │
│   POST /api/v1/verify-otp                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                     ┌─────────┴──────────┐
                     │                    │
                     ▼                    ▼
          ┌──────────────────┐  ┌──────────────────────┐
          │ Feature Extractor│  │  Behavioral Engine    │
          │ (33 features)    │  │  (baseline compare)   │
          │ - 16 Tabular     │  │  - typing speed       │
          │ - 4 ATLAS        │  │  - phone angle        │
          │ - 13 Behavioral  │  │  - swipe speed        │
          └────────┬─────────┘  └──────────┬───────────┘
                   │                       │
                   └──────────┬────────────┘
                              │
                              ▼
               ┌──────────────────────────┐
               │    XGBoost Classifier    │
               │   98%+ Accuracy          │
               │   SMOTE Balanced         │
               └──────────────┬───────────┘
                              │
                              ▼
               ┌──────────────────────────┐
               │   Decision Engine        │
               ├──────────────────────────┤
               │  prob > 0.80 → SUSPEND   │
               │  prob > 0.50 → OTP       │
               │  prob < 0.50 → ALLOW     │
               └──────────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         ┌─────────┐   ┌───────────┐   ┌──────────────┐
         │  ALLOW  │   │OTP_REQUIRED│  │   SUSPEND     │
         │ ✅      │   │ ⚠️ SMS OTP │   │ 🚨 OTP + SMS │
         └─────────┘   └─────┬─────┘   └──────┬───────┘
                             │                 │
                             ▼                 ▼
                      ┌─────────────────────────────┐
                      │  OTP Verification            │
                      ├─────────────────────────────┤
                      │  Valid  → UNLOCK             │
                      │  Invalid→ BLOCK + BLACKLIST  │
                      └─────────────────────────────┘

Blacklisting (only if fraud confirmed, prob > 0.90):
  - IP Address → ip_blacklist.json
  - Device ID  → device_blacklist.json
  - Beneficiary→ beneficiary_blacklist.json
```

## Streaming Architecture (Kafka/Redis)

```
Transaction Source
      │
      ▼
Kafka Producer ──────────► Kafka Topic: ato_transactions
                                   │
                                   ▼
                         ATO Consumer (background)
                                   │
                                   ▼
                         detect_ato() → result
                                   │
                          ┌────────┴────────┐
                    (ALLOW)│          (ALERT)│
                          │                 ▼
                          │      Kafka Topic: ato_alerts
                          │                 │
                          ▼                 ▼
                       (ignore)       Admin Dashboard
```

## Data Pipeline

```
raw_data/*.csv
      │
      ▼
data_processor.py
      │ (schema normalization, type coercion, synthetic fallback)
      ▼
data/ato_dataset_processed.csv
      │
      ▼
smote_augmentation.py
      │ (40% fraud ratio target)
      ▼
data/ato_dataset_with_smote.csv
      │
      ▼
model.py (XGBoost training)
      │
      ▼
models/ato_xgboost_model.json
```

## SMOTE Usage: Benchmarking vs. Production

SMOTE (Synthetic Minority Over-sampling) is used **only during benchmarking** to balance the dataset for accurate performance metrics. In **production/streaming**, the model operates on real unbalanced data.

### 📊 Benchmarking Mode (with SMOTE)
```bash
python scripts/train_model.py --benchmark
```
- **When**: Testing, comparing algorithms, establishing baseline performance
- **Dataset**: SMOTE-augmented (40% fraud ratio)
- **Use Case**: Report metrics like accuracy, F1, AUC-ROC
- **Output**: Metrics reflect balanced dataset performance

### 🚀 Production Mode (without SMOTE)
```bash
python scripts/train_model.py
```
- **When**: Deploying for real-time transaction detection
- **Dataset**: Raw processed data (real fraud ratio ~1-5%)
- **Use Case**: Catch fraudulent transactions as they arrive in streaming
- **Output**: Model trains on realistic class distribution

**Note**: The trained model (.json) is identical; only the training data differs. Once deployed, the model catches fraud on live streaming data without any augmentation.
