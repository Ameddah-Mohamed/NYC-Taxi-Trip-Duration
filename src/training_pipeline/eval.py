"""
Model Evaluation Module on Holdout Split for NYC Taxi Pipeline.

- Loads feature-engineered holdout dataset (unseen test split from June 2016).
- Loads trained model artifact.
- Filters holdout targets to the training distribution (100 < duration <= 4000)
  for official metrics, so data-entry errors (e.g. 24h durations) kept in the
  raw holdout split don't dominate RMSE. Unfiltered robustness stats are
  reported alongside for transparency.
- Applies the same 100s prediction floor as the inference pipeline.
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
    min_duration: int = settings.min_trip_duration,
    max_duration: int = settings.max_trip_duration,
) -> Dict[str, float]:
    """Evaluate trained model on holdout set and save summary metrics.

    Official metrics are computed on the training-distribution slice
    (min_duration < trip_duration <= max_duration). Rows outside that range
    are data-entry outliers present only in the raw holdout split (the
    training/eval splits are filtered upstream in preprocess.py).
    """
    h_path = Path(holdout_path) if holdout_path is not None else settings.processed_dir / "feature_engineered_holdout.csv"
    m_path = Path(model_path) if model_path is not None else settings.best_model_path
    out_json = Path(output_metrics_path) if output_metrics_path is not None else settings.models_dir / "eval_metrics.json"

    if not m_path.exists():
        # Fallback to baseline if best_model has not been tuned yet
        m_path = settings.baseline_model_path

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at {m_path}. Train a model first.")

    print(f"Loading holdout data from {h_path}...")
    df = pd.read_csv(h_path)
    target = "trip_duration"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in holdout dataset.")

    total_samples = int(len(df))

    # Like-for-like filter: match the training distribution so a handful of
    # corrupt rows (e.g. 86387s / 24h durations) can't dominate squared error.
    in_range = (df[target] > min_duration) & (df[target] <= max_duration)
    excluded = int((~in_range).sum())
    df_eval = df[in_range].reset_index(drop=True)
    if df_eval.empty:
        raise ValueError(
            f"No holdout rows in training range ({min_duration}s, {max_duration}s]; "
            f"total rows: {total_samples}."
        )
    if excluded:
        print(f"   Excluded {excluded}/{total_samples} holdout outliers outside "
              f"({min_duration}s, {max_duration}s] from official metrics.")

    X_holdout = df_eval.drop(columns=[target])
    y_holdout = df_eval[target]

    print(f"Loading model from {m_path}...")
    model = load(m_path)

    raw_preds = model.predict(X_holdout)
    # Same physical floor as inference_pipeline.inference.predict()
    preds = np.maximum(raw_preds, settings.min_trip_duration)

    rmse = float(root_mean_squared_error(y_holdout, preds))
    mae = float(mean_absolute_error(y_holdout, preds))
    med_ae = float(median_absolute_error(y_holdout, preds))
    r2 = float(r2_score(y_holdout, preds))

    # Error percentage relative to mean duration
    mean_actual = float(y_holdout.mean())
    mape_proxy = float((mae / mean_actual) * 100)

    # Unfiltered robustness stats on the full holdout (floored preds).
    X_full = df.drop(columns=[target])
    y_full = df[target]
    full_preds = np.maximum(model.predict(X_full), settings.min_trip_duration)
    unfiltered_rmse = float(root_mean_squared_error(y_full, full_preds))
    unfiltered_mae = float(mean_absolute_error(y_full, full_preds))

    metrics = {
        "holdout_samples": int(len(df_eval)),
        "holdout_total_samples": total_samples,
        "holdout_excluded_outliers": excluded,
        "duration_filter_seconds": [min_duration, max_duration],
        "rmse_seconds": round(rmse, 2),
        "mae_seconds": round(mae, 2),
        "median_absolute_error_seconds": round(med_ae, 2),
        "r2_score": round(r2, 4),
        "mean_actual_duration_seconds": round(mean_actual, 2),
        "error_percentage": round(mape_proxy, 2),
        "holdout_unfiltered_rmse_seconds": round(unfiltered_rmse, 2),
        "holdout_unfiltered_mae_seconds": round(unfiltered_mae, 2),
    }

    print("Holdout Evaluation Results (official: filtered to training range):")
    print(f"   Samples Evaluated: {metrics['holdout_samples']} (total {total_samples}, excluded {excluded})")
    print(f"   RMSE:              {metrics['rmse_seconds']} s")
    print(f"   MAE:               {metrics['mae_seconds']} s")
    print(f"   Median Abs Error:  {metrics['median_absolute_error_seconds']} s")
    print(f"   R² Score:          {metrics['r2_score']}")
    print(f"   Avg Error %:       {metrics['error_percentage']} %")
    print(f"   Unfiltered (robustness): RMSE {metrics['holdout_unfiltered_rmse_seconds']} s, "
          f"MAE {metrics['holdout_unfiltered_mae_seconds']} s")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_json}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
