"""Leakage-free benchmark for the generated behavioral ATO dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from behavioral_dataset_generator import generate_behavioral_dataset
from live_behavioral_model import LIVE_METADATA_PATH, LIVE_PIPELINE_PATH
from utils import DATA_DIR, MODELS_DIR, RAW_DATA_DIR


DROP_FEATURES = {
    "fraud_bool",
    "is_ato",
    "device_fingerprint_canvas",
    "device_fingerprint_webgl",
    "user_agent_raw",
    "user_agent_parsed",
    "network_connection_type",
    "device_os",
    "source",
    "payment_type",
}


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if c not in DROP_FEATURES]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]


def run_benchmark(
    dataset_path: Path,
    rows_if_generate: int,
    noise_rate: float,
    label_noise_rate: float,
    test_size: float,
    random_state: int,
) -> dict:
    if not dataset_path.exists():
        generate_behavioral_dataset(
            n=rows_if_generate,
            noise_rate=noise_rate,
            label_noise_rate=label_noise_rate,
            output_path=dataset_path,
            random_state=random_state,
        )

    df = pd.read_csv(dataset_path)
    target_col = "is_ato" if "is_ato" in df.columns else "fraud_bool"
    feature_cols = _numeric_feature_columns(df)
    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_smote, y_train_smote, eval_set=[(X_test_scaled, y_test)], verbose=False)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    balanced_train_path = DATA_DIR / "ato_behavioral_train_smote_50_50.csv"
    test_path = DATA_DIR / "ato_behavioral_test_holdout.csv"
    model_path = MODELS_DIR / "ato_behavioral_xgboost_benchmark.json"
    metrics_path = MODELS_DIR / "behavioral_benchmark_metrics.json"

    X_train_smote_unscaled = scaler.inverse_transform(X_train_smote)
    balanced_train = pd.DataFrame(X_train_smote_unscaled, columns=feature_cols)
    balanced_train[target_col] = np.asarray(y_train_smote, dtype=int)
    balanced_train.to_csv(balanced_train_path, index=False)

    test_out = X_test.copy()
    test_out[target_col] = y_test.to_numpy()
    test_out.to_csv(test_path, index=False)

    model.save_model(str(model_path))
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "pipeline": Pipeline(
                [
                    ("imputer", imputer),
                    ("scaler", scaler),
                    ("model", model),
                ]
            ),
        },
        LIVE_PIPELINE_PATH,
    )

    metrics = {
        "dataset_path": str(dataset_path),
        "raw_rows": int(len(df)),
        "raw_fraud_rate": round(float(y.mean()), 4),
        "feature_count": len(feature_cols),
        "features_used": feature_cols,
        "test_size": test_size,
        "train_rows_before_smote": int(len(y_train)),
        "train_rows_after_smote": int(len(y_train_smote)),
        "test_rows": int(len(y_test)),
        "train_distribution_after_smote": {
            "real": int((y_train_smote == 0).sum()),
            "fraud": int((y_train_smote == 1).sum()),
            "fraud_rate": round(float(np.mean(y_train_smote)), 4),
        },
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision_fraud": round(float(precision_score(y_test, y_pred)), 4),
        "recall_fraud": round(float(recall_score(y_test, y_pred)), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "true_positive_rate": round(float(tp / max(tp + fn, 1)), 4),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "balanced_train_csv": str(balanced_train_path),
        "holdout_test_csv": str(test_path),
        "model_path": str(model_path),
        "live_pipeline_path": str(LIVE_PIPELINE_PATH),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(LIVE_METADATA_PATH, "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "metrics_path": str(metrics_path),
                "trained_from": str(dataset_path),
                "train_rows_after_smote": int(len(y_train_smote)),
                "test_rows": int(len(y_test)),
            },
            f,
            indent=2,
            default=str,
        )

    metrics["metrics_path"] = str(metrics_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run leakage-free XGBoost benchmark on behavioral ATO data")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RAW_DATA_DIR / "ato_behavioral_dataset.csv",
        help="Input dataset. Generated if it does not exist.",
    )
    parser.add_argument("--rows", type=int, default=50_000, help="Rows to generate if input is missing")
    parser.add_argument("--noise-rate", type=float, default=0.15, help="Noise rate used only if generating missing input")
    parser.add_argument("--label-noise-rate", type=float, default=0.01, help="Label noise used only if generating missing input")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = run_benchmark(
        dataset_path=args.dataset,
        rows_if_generate=args.rows,
        noise_rate=args.noise_rate,
        label_noise_rate=args.label_noise_rate,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("=" * 70)
    print("Leakage-free Behavioral XGBoost Benchmark")
    print("=" * 70)
    print(f"Raw rows:                 {results['raw_rows']:,}")
    print(f"Raw fraud rate:           {results['raw_fraud_rate'] * 100:.2f}%")
    print(f"Train before SMOTE:       {results['train_rows_before_smote']:,}")
    print(f"Train after SMOTE:        {results['train_rows_after_smote']:,}")
    print(
        "SMOTE train balance:      "
        f"{results['train_distribution_after_smote']['fraud_rate'] * 100:.2f}% fraud"
    )
    print(f"Holdout test rows:        {results['test_rows']:,}")
    print(f"Accuracy:                 {results['accuracy'] * 100:.2f}%")
    print(f"AUC-ROC:                  {results['auc_roc']:.4f}")
    print(f"F1 Score:                 {results['f1_score']:.4f}")
    print(f"Precision (Fraud):        {results['precision_fraud']:.4f}")
    print(f"Recall (Fraud):           {results['recall_fraud']:.4f}")
    print(f"False Positive Rate:      {results['false_positive_rate'] * 100:.2f}%")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]: {results['confusion_matrix']}")
    print(f"Balanced train CSV:       {results['balanced_train_csv']}")
    print(f"Holdout test CSV:         {results['holdout_test_csv']}")
    print(f"Metrics JSON:             {results['metrics_path']}")
    print(f"Model:                    {results['model_path']}")
    print(f"Live pipeline:            {results['live_pipeline_path']}")
