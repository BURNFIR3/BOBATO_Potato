"""
src/model.py
XGBoost model training with hyperparameter tuning for ATO detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import DATA_DIR, MODELS_DIR, setup_logger
from feature_extractor import MODEL_FEATURES

setup_logger("ato_detection")


class ATOModel:
    """Wrapper around XGBClassifier for ATO detection with hyperparameter tuning."""

    MODEL_PATH = MODELS_DIR / "ato_xgboost_model.json"
    METRICS_PATH = MODELS_DIR / "training_metrics.json"
    HPARAMS_PATH = MODELS_DIR / "best_hparams.json"

    def __init__(self) -> None:
        self._model = None

    # ── Hyperparameter Tuning ──────────────────────────────────────────────
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Find best hyperparameters using Optuna.
        Returns best params dict.
        """
        try:
            import optuna
        except ImportError:
            logger.warning(
                "optuna not installed. Using default hyperparameters.")
            return self._get_default_params(y_train)

        logger.info("🔍 Starting hyperparameter tuning with Optuna...")

        def objective(trial):
            # Hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.5),
            }

            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise

            scale_pos_weight = int(
                (y_train == 0).sum() / max((y_train == 1).sum(), 1))
            model = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
            )

            # Cross-validation score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        # Optimization
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        best_params = study.best_params
        best_params['scale_pos_weight'] = int(
            (y_train == 0).sum() / max((y_train == 1).sum(), 1))

        logger.success(f"✓ Best AUC-ROC: {study.best_value:.4f}")
        logger.info(f"  Best params: {best_params}")

        return best_params

    def _get_default_params(self, y_train: pd.Series) -> dict:
        """Default XGBoost parameters."""
        scale_pos_weight = int((y_train == 0).sum() /
                               max((y_train == 1).sum(), 1))
        return {
            'n_estimators': 400,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos_weight,
        }

    # ── Training ───────────────────────────────────────────────────────────
    def train(
        self,
        dataset_path: str | Path | None = None,
        test_size: float = 0.20,
        random_state: int = 42,
        use_smote: bool = False,
        smote_sampling_strategy: float = 1.0,
        optimize_hyperparams: bool = True,
    ) -> dict[str, Any]:
        """
        Train an XGBoost classifier on the dataset with optional hyperparameter tuning.

        Args:
            dataset_path: Path to dataset. If None, uses processed data.
            test_size: Fraction of data for testing.
            random_state: Random seed for reproducibility.
            use_smote: If True, apply SMOTE to the training split only.
            smote_sampling_strategy: SMOTE fraud:real ratio for the training split.
            optimize_hyperparams: If True, tune hyperparameters with Optuna.

        Returns:
            Metrics dict.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            raise

        # ── Load dataset ──────────────────────────────────────────────────
        if dataset_path is None:
            dataset_path = DATA_DIR / "ato_dataset_processed.csv"
            logger.info("Loading processed dataset for training")

        logger.info(f"Loading data from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Keep only model features that exist
        available = [c for c in MODEL_FEATURES if c in df.columns]
        missing = set(MODEL_FEATURES) - set(available)
        if missing:
            logger.warning(f"Missing features: {missing}")

        X = df[available].astype(float)
        y = df["is_ato"].astype(int)

        logger.info(
            f"Dataset: {len(X):,} rows | Fraud: {y.sum():,} ({y.mean()*100:.2f}%)")
        logger.info(f"Features: {len(available)}")

        # ── Train / test split ────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        train_samples_before_smote = len(X_train)
        train_distribution_before_smote = {
            "real": int((y_train == 0).sum()),
            "fraud": int((y_train == 1).sum()),
            "fraud_rate": float(y_train.mean()),
        }

        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
            except ImportError:
                logger.error("imbalanced-learn not installed. Run: pip install imbalanced-learn")
                raise

            fraud_count = int((y_train == 1).sum())
            if fraud_count < 2:
                logger.warning("Skipping SMOTE because the training fold has fewer than 2 fraud rows.")
            else:
                k_neighbors = min(5, fraud_count - 1)
                smote = SMOTE(
                    sampling_strategy=smote_sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(
                    "Applied SMOTE to training split only: "
                    f"{train_samples_before_smote:,} -> {len(X_train):,} rows | "
                    f"Fraud: {int((y_train == 1).sum()):,} ({y_train.mean()*100:.2f}%)"
                )

        # ── Get hyperparameters ───────────────────────────────────────────
        if optimize_hyperparams:
            params = self._tune_hyperparameters(X_train, y_train)
        else:
            params = self._get_default_params(y_train)

        # ── Train XGBoost ─────────────────────────────────────────────────
        logger.info("Training XGBoost model...")
        self._model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Evaluate ──────────────────────────────────────────────────────
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)

        metrics = {
            "accuracy": round(acc, 4),
            "auc_roc": round(auc, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positive_rate": round(tpr, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "train_samples": len(X_train),
            "train_samples_before_smote": train_samples_before_smote,
            "test_samples": len(X_test),
            "smote_applied_to_train_only": use_smote,
            "smote_sampling_strategy": smote_sampling_strategy if use_smote else None,
            "train_distribution_before_smote": train_distribution_before_smote,
            "train_distribution_after_smote": {
                "real": int((y_train == 0).sum()),
                "fraud": int((y_train == 1).sum()),
                "fraud_rate": float(y_train.mean()),
            },
            "hyperparameters_optimized": optimize_hyperparams,
        }

        # ── Save model ────────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self.MODEL_PATH))
        with open(self.METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        with open(self.HPARAMS_PATH, "w") as f:
            json.dump(params, f, indent=2, default=str)

        logger.success(f"✓ Model saved → {self.MODEL_PATH}")
        logger.info(
            f"  Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | F1: {f1:.4f}")

        # Log precision/recall for fraud class (key may be '1' or '1.0')
        fraud_key = '1' if '1' in report else '1.0'
        if fraud_key in report:
            logger.info(
                f"  Precision: {report[fraud_key]['precision']:.4f} | Recall: {report[fraud_key]['recall']:.4f}")

        return metrics

    # ── Inference ──────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load a previously trained model."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed")

        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {self.MODEL_PATH}. Run: python scripts/train_model.py"
            )
        self._model = XGBClassifier()
        self._model.load_model(str(self.MODEL_PATH))
        logger.info(f"Model loaded from {self.MODEL_PATH}")

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return ATO probability for a single feature dict."""
        if self._model is None:
            self.load()
        X = pd.DataFrame([features])[MODEL_FEATURES].astype(float)
        prob = float(self._model.predict_proba(X)[0, 1])
        return prob

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability array for a batch DataFrame."""
        if self._model is None:
            self.load()
        available = [c for c in MODEL_FEATURES if c in df.columns]
        missing = set(MODEL_FEATURES) - set(available)
        for col in missing:
            df[col] = 0.0
        X = df[MODEL_FEATURES].astype(float)
        return self._model.predict_proba(X)[:, 1]

    def get_training_metrics(self) -> dict:
        if self.METRICS_PATH.exists():
            with open(self.METRICS_PATH) as f:
                return json.load(f)
        return {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Singleton
ato_model = ATOModel()

if __name__ == "__main__":
    metrics = ato_model.train(use_smote=False)
    print(f"\n{'='*50}")
    print("Training complete! (Production mode - no SMOTE)")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  FPR:      {metrics['false_positive_rate']*100:.2f}%")
