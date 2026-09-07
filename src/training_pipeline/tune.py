"""
Hyperparameter Optimization & Model Registry Module for NYC Taxi Pipeline.

- Supports Optuna Bayesian tuning for both LightGBM and XGBoost.
- Logs each trial as a nested child run in MLflow.
- Retrains the best model on full training set.
- Registers winning model into the MLflow Model Registry.
- Saves serialized artifact to models/best_model.pkl.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from joblib import dump
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from src.config import settings
from src.training_pipeline.train import load_train_eval_data

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_lightgbm_params(trial: optuna.Trial, random_state: int) -> Dict[str, Any]:
    """Optuna hyperparameter search space tailored for LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 450),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def _suggest_xgboost_params(trial: optuna.Trial, random_state: int) -> Dict[str, Any]:
    """Optuna hyperparameter search space tailored for XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }


def tune_model(
    model_type: str = "lightgbm",
    train_path: Union[Path, str, None] = None,
    eval_path: Union[Path, str, None] = None,
    model_output: Union[Path, str, None] = None,
    n_trials: int = 10,
    sample_frac: Optional[float] = None,
    random_state: int = settings.random_state,
    experiment_name: str = settings.mlflow_experiment_name,
    model_registry_name: str = settings.mlflow_model_name,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run Optuna tuning, log trials to MLflow, and register best model in Model Registry."""
    model_type = model_type.lower()
    if model_type not in ["lightgbm", "xgboost"]:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'lightgbm' or 'xgboost'.")

    out_path = Path(model_output) if model_output is not None else settings.best_model_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_eval, y_eval = load_train_eval_data(
        train_path=train_path,
        eval_path=eval_path,
        sample_frac=sample_frac,
        random_state=random_state,
    )

    print(f"🎯 Starting Optuna study for {model_type.upper()} ({n_trials} trials)...")

    parent_run_name = f"optuna_{model_type}_{n_trials}_trials"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.set_tag("pipeline", "tuning")
        mlflow.set_tag("model_type", model_type)
        mlflow.log_param("n_trials", n_trials)
        if sample_frac:
            mlflow.log_param("sample_frac", sample_frac)

        def objective(trial: optuna.Trial) -> float:
            if model_type == "lightgbm":
                params = _suggest_lightgbm_params(trial, random_state)
                model = LGBMRegressor(**params)
            else:
                params = _suggest_xgboost_params(trial, random_state)
                model = XGBRegressor(**params)

            with mlflow.start_run(nested=True, run_name=f"{model_type}_trial_{trial.number}"):
                model.fit(X_train, y_train)
                preds = model.predict(X_eval)

                rmse = float(root_mean_squared_error(y_eval, preds))
                mae = float(mean_absolute_error(y_eval, preds))
                r2 = float(r2_score(y_eval, preds))

                mlflow.log_params(params)
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params
        print(f"🏆 Best {model_type.upper()} Trial #{study.best_trial.number} RMSE: {study.best_value:.2f}s")
        print(f"   Best params: {best_params}")

        # Retrain best model
        print(f"🔨 Retraining best {model_type.upper()} model with optimal parameters...")
        if model_type == "lightgbm":
            final_params = {**best_params, "random_state": random_state, "n_jobs": -1, "verbose": -1}
            best_model = LGBMRegressor(**final_params)
        else:
            final_params = {**best_params, "tree_method": "hist", "random_state": random_state, "n_jobs": -1}
            best_model = XGBRegressor(**final_params)

        best_model.fit(X_train, y_train)

        # Evaluate best model
        y_pred = best_model.predict(X_eval)
        best_metrics = {
            "rmse": float(root_mean_squared_error(y_eval, y_pred)),
            "mae": float(mean_absolute_error(y_eval, y_pred)),
            "r2": float(r2_score(y_eval, y_pred)),
        }

        # Log best parameters and metrics
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(best_metrics)

        # Log model artifact and register
        print(f"📦 Registering winning {model_type.upper()} model in MLflow Model Registry ({model_registry_name})...")
        if model_type == "lightgbm":
            mlflow.lightgbm.log_model(
                lgb_model=best_model,
                artifact_path="model",
                registered_model_name=model_registry_name,
            )
        else:
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                artifact_path="model",
                registered_model_name=model_registry_name,
            )

        # Tag version as Production
        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_registry_name)
        if latest_versions:
            current_ver = latest_versions[-1].version
            client.set_model_version_tag(
                name=model_registry_name,
                version=current_ver,
                key="stage",
                value="Production",
            )
            client.set_model_version_tag(
                name=model_registry_name,
                version=current_ver,
                key="model_type",
                value=model_type,
            )
            print(f"🏷️ Tagged {model_registry_name} v{current_ver} ({model_type}) as Production")

    # Persist locally to models/best_model.pkl
    dump(best_model, out_path)
    print(f"💾 Saved best model to {out_path}")
    print(f"📊 Best Tuned {model_type.upper()} Performance:")
    print(f"   RMSE: {best_metrics['rmse']:.2f} seconds")
    print(f"   MAE:  {best_metrics['mae']:.2f} seconds")
    print(f"   R²:   {best_metrics['r2']:.4f}")

    return best_params, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune model with Optuna and MLflow.")
    parser.add_argument("--model_type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"], help="Model type to tune")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    args = parser.parse_args()
    tune_model(model_type=args.model_type, n_trials=args.n_trials)
