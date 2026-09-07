"""
Model Evaluation Module on Holdout Split for NYC Taxi Pipeline.

- Loads feature-engineered holdout dataset (unseen test split from June 2016).
- Loads trained model artifact.
- Generates comprehensive evaluation metrics (RMSE, MAE, R2, Median Absolute Error).
- Exports metrics summary to models/eval_metrics.json.
"""

import json
from pathlib import Path
from typing import Dict, Union
from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

from src.config import settings


def evaluate_model(
    holdout_path: Union[Path, str, None] = None,
    model_path: Union[Path, str, None] = None,
    output_metrics_path: Union[Path, str, None] = None,
) -> Dict[str, float]:
    """Evaluate trained model on holdout set and save summary metrics."""
    h_path = Path(holdout_path) if holdout_path is not None else settings.processed_dir / "feature_engineered_holdout.csv"
    m_path = Path(model_path) if model_path is not None else settings.best_model_path
    out_json = Path(output_metrics_path) if output_metrics_path is not None else settings.models_dir / "eval_metrics.json"

    if not m_path.exists():
        # Fallback to baseline if best_model has not been tuned yet
        m_path = settings.baseline_model_path

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at {m_path}. Train a model first.")

    print(f"📖 Loading holdout data from {h_path}...")
    df = pd.read_csv(h_path)
    target = "trip_duration"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in holdout dataset.")

    X_holdout = df.drop(columns=[target])
    y_holdout = df[target]

    print(f"📦 Loading model from {m_path}...")
    model = load(m_path)

    preds = model.predict(X_holdout)

    rmse = float(root_mean_squared_error(y_holdout, preds))
    mae = float(mean_absolute_error(y_holdout, preds))
    med_ae = float(median_absolute_error(y_holdout, preds))
    r2 = float(r2_score(y_holdout, preds))

    # Error percentage relative to mean duration
    mean_actual = float(y_holdout.mean())
    mape_proxy = float((mae / mean_actual) * 100)

    metrics = {
        "holdout_samples": int(len(df)),
        "rmse_seconds": round(rmse, 2),
        "mae_seconds": round(mae, 2),
        "median_absolute_error_seconds": round(med_ae, 2),
        "r2_score": round(r2, 4),
        "mean_actual_duration_seconds": round(mean_actual, 2),
        "error_percentage": round(mape_proxy, 2),
    }

    print("🏁 Holdout Evaluation Results:")
    print(f"   Samples Evaluated: {metrics['holdout_samples']}")
    print(f"   RMSE:              {metrics['rmse_seconds']} s")
    print(f"   MAE:               {metrics['mae_seconds']} s")
    print(f"   Median Abs Error:  {metrics['median_absolute_error_seconds']} s")
    print(f"   R² Score:          {metrics['r2_score']}")
    print(f"   Avg Error %:       {metrics['error_percentage']} %")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {out_json}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
