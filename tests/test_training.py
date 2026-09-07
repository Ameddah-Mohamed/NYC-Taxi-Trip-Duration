"""Integration tests for training and evaluation pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.train import train_model


@pytest.fixture
def synthetic_fe_data(tmp_path):
    """Create lightweight synthetic feature-engineered datasets for test isolation."""
    cols = [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "pickup_hour", "pickup_dayofweek", "distance_km",
        "vendor_id_1", "vendor_id_2",
        "pickup_month_1", "pickup_month_2", "pickup_month_3", "pickup_month_4",
        "trip_duration"
    ]
    np.random.seed(42)
    n_rows = 50
    data = np.random.randn(n_rows, len(cols) - 1)
    durations = np.random.uniform(200, 1500, size=(n_rows, 1))
    full_data = np.hstack([data, durations])

    df = pd.DataFrame(full_data, columns=cols)
    train_file = tmp_path / "test_fe_train.csv"
    eval_file = tmp_path / "test_fe_eval.csv"
    holdout_file = tmp_path / "test_fe_holdout.csv"

    df.to_csv(train_file, index=False)
    df.to_csv(eval_file, index=False)
    df.to_csv(holdout_file, index=False)

    return train_file, eval_file, holdout_file


def test_baseline_training_fast_sample(tmp_path, synthetic_fe_data):
    train_file, eval_file, _ = synthetic_fe_data
    temp_model_path = tmp_path / "temp_baseline.pkl"

    model, metrics = train_model(
        model_type="lightgbm",
        train_path=train_file,
        eval_path=eval_file,
        model_output=temp_model_path,
        log_to_mlflow=False,
    )
    assert temp_model_path.exists()
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_eval_metrics_output(tmp_path, synthetic_fe_data):
    _, _, holdout_file = synthetic_fe_data
    metrics_path = tmp_path / "test_metrics.json"

    metrics = evaluate_model(
        holdout_path=holdout_file,
        output_metrics_path=metrics_path,
    )
    assert metrics_path.exists()
    assert "rmse_seconds" in metrics
    assert "median_absolute_error_seconds" in metrics
    assert metrics["holdout_samples"] > 0
