"""scripts/train_model.py - Train XGBoost model with optional hyperparameter tuning

Usage:
  python scripts/train_model.py              # Production mode (no SMOTE)
  python scripts/train_model.py --benchmark  # Benchmark mode (SMOTE on train split only)
  python scripts/train_model.py --optimize   # Optimize hyperparameters with Optuna
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to python path BEFORE importing model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ato_model
from utils import DATA_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train ATO detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py              # Production mode (processed data)
  python scripts/train_model.py --benchmark  # Benchmark mode (SMOTE on train split only)
  python scripts/train_model.py --optimize   # Optimize hyperparameters
        """,
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize hyperparameters using Optuna",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Use SMOTE on the training split for benchmarking",
    )
    args = parser.parse_args()

    mode = "BENCHMARK (train-only SMOTE 50/50)" if args.benchmark else "PRODUCTION (Real distribution)"
    if args.optimize:
        mode += " + HYPERPARAMETER OPTIMIZATION"
        
    dataset_path = DATA_DIR / "ato_dataset_processed.csv"

    print("=" * 70)
    print(f"Training XGBoost ATO Detection Model")
    print(f"Mode: {mode}")
    print("=" * 70)

    metrics = ato_model.train(
        dataset_path=dataset_path,
        use_smote=args.benchmark,
        smote_sampling_strategy=1.0,
        optimize_hyperparams=args.optimize
    )

    print(f"\n{'='*70}")
    print("[DONE] Training Complete!")
    print(f"{'='*70}")
    print(f"Mode:                    {mode}")
    print(f"Accuracy:                {metrics['accuracy']*100:.2f}%")
    print(f"AUC-ROC:                 {metrics['auc_roc']:.4f}")
    print(f"F1 Score:                {metrics['f1_score']:.4f}")
    if 'classification_report' in metrics:
        fraud_key = '1' if '1' in metrics['classification_report'] else '1.0'
        if fraud_key in metrics['classification_report']:
            print(f"Precision (Fraud):       {metrics['classification_report'][fraud_key]['precision']:.4f}")
            print(f"Recall (Fraud):          {metrics['classification_report'][fraud_key]['recall']:.4f}")
    print(f"Test Samples:            {metrics['test_samples']:,}")
    print(f"Train Samples:           {metrics['train_samples']:,}")
    if metrics.get("smote_applied_to_train_only"):
        before = metrics["train_samples_before_smote"]
        dist = metrics["train_distribution_after_smote"]
        print(f"Train Before SMOTE:      {before:,}")
        print(f"Train SMOTE Balance:     {dist['fraud']:,} fraud / {dist['real']:,} real")
    print(f"{'='*70}")
