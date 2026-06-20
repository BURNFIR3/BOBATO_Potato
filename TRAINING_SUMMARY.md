# Model Training Summary

## Data Processing

### Raw Dataset
- **Source**: `raw_data/Base.csv`
- **Size**: 1,000,000 transactions
- **Fraud Rate**: 1.10% (11,029 fraud cases)
- **Features**: 31 columns (30 features + 1 target)

### Features Used
- **Numeric Features**: 26
  - Income, Name-Email Similarity, Address Months Count
  - Customer Age, Days Since Request, Balance Amount
  - Zip Code Count, Velocity Metrics (6h, 24h, 4w)
  - Bank Branch Count, Email Distinctness
  - Credit Risk Score, Phone Validity
  - Bank Account Months, Other Cards
  - Proposed Credit Limit, Foreign Request Flag
  - Session Length, Keep-Alive Flag
  - Device Distinctness, Device Fraud Count
  - Transaction Month

- **Categorical Features**: 5 (Label Encoded)
  - Payment Type
  - Employment Status
  - Housing Status
  - Source
  - Device OS

## Model Training

### Architecture
- **Algorithm**: XGBoost Classifier
- **Scale Pos Weight**: 89 (automatically calculated to handle imbalance)
- **Train/Test Split**: 80/20
- **Framework**: scikit-learn compatible

### Hyperparameters (Optimized)
```json
{
  "n_estimators": 400,
  "max_depth": 7,
  "learning_rate": 0.05,
  "subsample": 0.85,
  "colsample_bytree": 0.85,
  "min_child_weight": 3,
  "gamma": 0.1,
  "reg_alpha": 0.1,
  "reg_lambda": 1.5,
  "scale_pos_weight": 89
}
```

### Training Configuration
- **Hyperparameter Tuning**: Available via Optuna (30 trials)
- **Cross-Validation**: 5-fold Stratified KFold
- **Scoring Metric**: ROC-AUC (maximized during tuning)
- **Optimization Framework**: Optuna (Tree-structured Parzen Estimator)

## Final Model Performance

### Test Set Metrics (200,000 samples)
| Metric | Value |
|--------|-------|
| **Accuracy** | 89.39% |
| **AUC-ROC** | 0.8901 |
| **F1 Score** | 0.1252 |
| **Precision** | 6.89% |
| **Recall** | 68.86% |

### What These Metrics Mean
- **Recall (68.86%)**: The model catches ~69% of actual fraud cases ✅
- **Precision (6.89%)**: When model flags a transaction, ~7% are actually fraud (expected with 1% base rate)
- **AUC-ROC (0.8901)**: Excellent discrimination between fraud and legitimate transactions
- **Accuracy (89.39%)**: High accuracy due to majority legitimate class

### Interpretation for Production
- **High Recall**: Catches most fraud cases - important for risk mitigation
- **Conservative Precision**: False positives acceptable for fraud detection (better to verify than miss fraud)
- **Strong AUC**: Model discriminates well across probability thresholds

## Saved Artifacts

### Model Files
1. **`models/ato_xgboost_model.json`** (2.1 MB)
   - Trained XGBoost classifier
   - Ready for production inference
   - Loaded automatically during predictions

2. **`models/training_metrics.json`**
   - Complete performance metrics
   - Confusion matrix, classification report
   - Train/test split information

3. **`models/best_hparams.json`**
   - Optimal hyperparameters used
   - Can be compared with future tuning runs

4. **`data/ato_dataset_processed.csv`** (1 GB)
   - Processed dataset with all 31 features + target
   - Ready for retraining or analysis

## Usage

### Use Default Hyperparameters (Fast)
```bash
python scripts/train_model.py
```

### Optimize Hyperparameters (Slow - 30+ minutes)
```bash
python scripts/train_model.py --optimize
```

### Full Pipeline
```bash
# 1. Process raw data
python scripts/process_dataset.py

# 2. Train model
python scripts/train_model.py

# 3. Start API (production)
python scripts/start_api.py

# 4. Start streaming (real-time fraud detection)
python scripts/start_stream.py
```

## Key Features

### ✅ Completed
- [x] Raw dataset processing (1M records)
- [x] Feature engineering from real data
- [x] XGBoost model training
- [x] Hyperparameter tuning framework
- [x] Model evaluation and metrics
- [x] Production-ready model export

### 🚀 Ready for Production
- Production model (`ato_xgboost_model.json`)
- REST API endpoints
- Real-time Kafka/Redis streaming
- Admin dashboard
- Blacklist management
- OTP verification

## Next Steps

1. **Deploy Model**: Copy `models/ato_xgboost_model.json` to production
2. **Test API**: Use `curl` or Postman to test endpoints
3. **Monitor Performance**: Track fraud catch rate and false positive rate
4. **Retrain Periodically**: Collect new fraud cases and retrain quarterly
5. **A/B Test**: Compare with existing fraud detection system
6. **Fine-tune Thresholds**: Adjust decision thresholds based on business requirements

## Model Monitoring

### Key Metrics to Track
- Fraud catch rate (Recall)
- False positive rate
- Average transaction response time
- Model inference latency
- Daily/hourly fraud volume

### Retraining Triggers
- Fraud catch rate drops below 60%
- Model not seen fraud case in 30 days
- New fraud pattern emerges
- Quarterly scheduled retraining

---

**Model Generated**: 2026-06-20
**Training Data**: 1,000,000 transactions (1.1% fraud)
**Status**: ✅ Production Ready
