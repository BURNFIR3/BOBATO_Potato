"""tests/test_model.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from model import ATOModel
from data_processor import _generate_synthetic_dataset


def test_train_and_predict():
    """End-to-end: generate data → train → predict single sample."""
    import pandas as pd
    from pathlib import Path
    import tempfile, shutil

    # Use a small synthetic dataset
    df = _generate_synthetic_dataset(n=500)

    # Write to a temp CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)

        model = ATOModel()
        metrics = model.train(dataset_path=csv_path)

    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.70   # at least 70% on small set


def test_predict_proba_range():
    """Predict proba must be in [0, 1]."""
    model = ATOModel()
    try:
        model.load()
    except FileNotFoundError:
        pytest.skip("Model not trained yet – skipping inference test")

    from feature_extractor import MODEL_FEATURES
    dummy = {k: 0.0 for k in MODEL_FEATURES}
    prob = model.predict_proba(dummy)
    assert 0.0 <= prob <= 1.0
