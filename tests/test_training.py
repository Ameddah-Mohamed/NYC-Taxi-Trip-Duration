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
    train_file, _, holdout_file = synthetic_fe_data
    temp_model_path = tmp_path / "temp_eval_model.pkl"
    metrics_path = tmp_path / "test_metrics.json"

    # Train a temporary model on the synthetic train file to guarantee shape match
    train_model(
        model_type="lightgbm",
        train_path=train_file,
        eval_path=train_file,
        model_output=temp_model_path,
        log_to_mlflow=False,
    )

    metrics = evaluate_model(
        holdout_path=holdout_file,
        model_path=temp_model_path,
        output_metrics_path=metrics_path,
    )
    assert metrics_path.exists()
    assert "rmse_seconds" in metrics
    assert "median_absolute_error_seconds" in metrics
    assert metrics["holdout_samples"] > 0


def test_eval_filters_outliers_and_reports_both(tmp_path, synthetic_fe_data):
    """Holdout outliers outside the training range must not drive official metrics."""
    import json

    train_file, _, _ = synthetic_fe_data
    temp_model_path = tmp_path / "temp_filter_model.pkl"
    metrics_path = tmp_path / "test_filter_metrics.json"

    train_model(
        model_type="lightgbm",
        train_path=train_file,
        eval_path=train_file,
        model_output=temp_model_path,
        log_to_mlflow=False,
    )

    # Build a holdout with 46 in-range rows + 4 extreme outliers
    base = pd.read_csv(train_file).head(46)
    outlier_data = {
        "pickup_longitude": [0.0, 0.0, 0.0, 0.0],
        "pickup_latitude": [0.0, 0.0, 0.0, 0.0],
        "dropoff_longitude": [0.0, 0.0, 0.0, 0.0],
        "dropoff_latitude": [0.0, 0.0, 0.0, 0.0],
        "pickup_hour": [0.0, 0.0, 0.0, 0.0],
        "pickup_dayofweek": [0.0, 0.0, 0.0, 0.0],
        "distance_km": [0.0, 0.0, 0.0, 0.0],
        "vendor_id_1": [1.0, 1.0, 1.0, 1.0],
        "vendor_id_2": [0.0, 0.0, 0.0, 0.0],
        "pickup_month_1": [1.0, 1.0, 1.0, 1.0],
        "pickup_month_2": [0.0, 0.0, 0.0, 0.0],
        "pickup_month_3": [0.0, 0.0, 0.0, 0.0],
        "pickup_month_4": [0.0, 0.0, 0.0, 0.0],
        "trip_duration": [5.0, 50.0, 80000.0, 86387.0],
    }
    outliers = pd.DataFrame(outlier_data)
    mixed = pd.concat([base, outliers], ignore_index=True)
    mixed_file = tmp_path / "test_fe_holdout_mixed.csv"
    mixed.to_csv(mixed_file, index=False)

    metrics = evaluate_model(
        holdout_path=mixed_file,
        model_path=temp_model_path,
        output_metrics_path=metrics_path,
    )

    # Official metrics cover only the in-range slice
    assert metrics["holdout_total_samples"] == 50
    assert metrics["holdout_excluded_outliers"] == 4
    assert metrics["holdout_samples"] == 46
    # Robustness stats preserve visibility into the full split
    assert metrics["holdout_unfiltered_rmse_seconds"] >= metrics["rmse_seconds"]
    assert metrics["r2_score"] > -10  # sane, not dominated by 24h outliers

    with open(metrics_path) as f:
        saved = json.load(f)
    assert saved["holdout_samples"] == 46
    assert saved["holdout_excluded_outliers"] == 4
