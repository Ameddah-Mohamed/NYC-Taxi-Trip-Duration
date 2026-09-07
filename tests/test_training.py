"""Integration tests for training and evaluation pipeline."""

from pathlib import Path
import pytest

from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.train import train_model


def test_baseline_training_fast_sample(tmp_path):
    # Train on 1% sample to run in ~0.5s without touching production model
    temp_model_path = tmp_path / "temp_baseline.pkl"
    model, metrics = train_model(
        model_type="lightgbm",
        sample_frac=0.01,
        model_output=temp_model_path,
        log_to_mlflow=False,
    )
    assert temp_model_path.exists()
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_eval_metrics_output(tmp_path):
    metrics_path = tmp_path / "test_metrics.json"
    metrics = evaluate_model(output_metrics_path=metrics_path)
    assert metrics_path.exists()
    assert "rmse_seconds" in metrics
    assert "median_absolute_error_seconds" in metrics
    assert metrics["holdout_samples"] > 0
