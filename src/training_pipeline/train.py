"""
Model Training Module for NYC Taxi Trip Duration Pipeline.

Supports training both LightGBM and XGBoost models with:
- Fast histogram / leaf-wise training
- Automated evaluation (RMSE, MAE, R2)
- MLflow experiment tracking
- Model artifact persistence
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from src.config import settings

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    """Optionally subsample DataFrame for fast verification or local tests."""
    if sample_frac is None or sample_frac <= 0 or sample_frac >= 1.0:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def load_train_eval_data(
    train_path: Union[Path, str, None] = None,
    eval_path: Union[Path, str, None] = None,
    sample_frac: Optional[float] = None,
    random_state: int = settings.random_state,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare X and y for train and eval splits."""
    t_path = Path(train_path) if train_path is not None else settings.processed_dir / "feature_engineered_train.csv"
    e_path = Path(eval_path) if eval_path is not None else settings.processed_dir / "feature_engineered_eval.csv"

    print(f"📖 Loading data: {t_path.name} and {e_path.name}")
    train_df = pd.read_csv(t_path)
    eval_df = pd.read_csv(e_path)

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "trip_duration"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    print(f"   X_train: {X_train.shape}, X_eval: {X_eval.shape}")
    return X_train, y_train, X_eval, y_eval


def train_model(
    model_type: str = "lightgbm",
    train_path: Union[Path, str, None] = None,
    eval_path: Union[Path, str, None] = None,
    model_output: Union[Path, str, None] = None,
    sample_frac: Optional[float] = None,
    random_state: int = settings.random_state,
    log_to_mlflow: bool = True,
    **custom_params: Any,
) -> Tuple[Any, Dict[str, float]]:
    """Train LightGBM or XGBoost model, evaluate metrics, and save artifact."""
    model_type = model_type.lower()
    if model_type not in ["lightgbm", "xgboost"]:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'lightgbm' or 'xgboost'.")

    out_path = Path(model_output) if model_output is not None else settings.baseline_model_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_eval, y_eval = load_train_eval_data(
        train_path=train_path,
        eval_path=eval_path,
        sample_frac=sample_frac,
        random_state=random_state,
    )

    if model_type == "lightgbm":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.08,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        default_params.update(custom_params)
        model = LGBMRegressor(**default_params)
    else:  # xgboost
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(custom_params)
        model = XGBRegressor(**default_params)

    print(f"🚀 Training {model_type.upper()} model with params: {default_params}")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_eval)
    rmse = float(root_mean_squared_error(y_eval, y_pred))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    print(f"📊 {model_type.upper()} Performance on Eval Set:")
    print(f"   RMSE: {rmse:.2f} seconds")
    print(f"   MAE:  {mae:.2f} seconds")
    print(f"   R²:   {r2:.4f}")

    # Persist model
    dump(model, out_path)
    print(f"💾 Saved {model_type} model to {out_path}")

    # Log to MLflow
    if log_to_mlflow:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"baseline_{model_type}"):
            mlflow.set_tag("model_type", model_type)
            mlflow.log_params(default_params)
            mlflow.log_metrics(metrics)
            if model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, artifact_path="model")
            else:
                mlflow.xgboost.log_model(model, artifact_path="model")
            print("📈 Logged run to MLflow")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model (LightGBM or XGBoost).")
    parser.add_argument("--model_type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"], help="Model type to train")
    args = parser.parse_args()
    train_model(model_type=args.model_type)
