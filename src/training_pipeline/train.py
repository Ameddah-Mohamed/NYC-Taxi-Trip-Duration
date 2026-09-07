"""
Baseline Model Training Module for NYC Taxi Trip Duration Pipeline.

- Reads feature-engineered train and eval CSVs.
- Trains an XGBoost baseline regressor using fast histogram method.
- Evaluates metrics (RMSE, MAE, R2).
- Logs experiment metrics to MLflow and persists baseline model artifact.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost

from src.config import settings

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    """Optionally subsample DataFrame for fast verification or local runs."""
    if sample_frac is None or sample_frac <= 0 or sample_frac >= 1.0:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def load_train_eval_data(
    train_path: Union[Path, str, None] = None,
    eval_path: Union[Path, str, None] = None,
    sample_frac: Optional[float] = None,
    random_state: int = settings.random_state,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare X and y for train and eval."""
    t_path = Path(train_path) if train_path is not None else settings.processed_dir / "feature_engineered_train.csv"
    e_path = Path(eval_path) if eval_path is not None else settings.processed_dir / "feature_engineered_eval.csv"

    print(f"📖 Loading data: {t_path} and {e_path}")
    train_df = pd.read_csv(t_path)
    eval_df = pd.read_csv(e_path)

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "trip_duration"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    print(f"   X_train: {X_train.shape}, X_eval: {X_eval.shape}")
    return X_train, y_train, X_eval, y_eval


def train_baseline_model(
    train_path: Union[Path, str, None] = None,
    eval_path: Union[Path, str, None] = None,
    model_output: Union[Path, str, None] = None,
    sample_frac: Optional[float] = None,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = settings.random_state,
    log_to_mlflow: bool = True,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """Train baseline XGBoost model and save artifact."""
    out_path = Path(model_output) if model_output is not None else settings.baseline_model_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_eval, y_eval = load_train_eval_data(
        train_path=train_path,
        eval_path=eval_path,
        sample_frac=sample_frac,
        random_state=random_state,
    )

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }

    print(f"🚀 Training baseline XGBoost model with params: {params}")
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_eval)
    rmse = float(root_mean_squared_error(y_eval, y_pred))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    print(f"📊 Baseline Model Performance on Eval:")
    print(f"   RMSE: {rmse:.2f} seconds")
    print(f"   MAE:  {mae:.2f} seconds")
    print(f"   R²:   {r2:.4f}")

    # Persist model
    dump(model, out_path)
    print(f"💾 Saved baseline model to {out_path}")

    # Log to MLflow
    if log_to_mlflow:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name="baseline_xgboost"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="model")
            print("📈 Logged baseline run to MLflow")

    return model, metrics


if __name__ == "__main__":
    train_baseline_model()
