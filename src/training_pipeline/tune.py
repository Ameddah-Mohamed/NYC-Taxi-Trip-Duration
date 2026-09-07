"""
Hyperparameter Optimization & Model Registry Module for NYC Taxi Pipeline.

- Conducts Bayesian optimization with Optuna.
- Logs every trial as a nested run inside MLflow.
- Retrains best model, logs model artifact, and registers in the MLflow Model Registry.
- Saves serialized artifact to models/best_model.pkl.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from joblib import dump
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from src.config import settings
from src.training_pipeline.train import load_train_eval_data

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(
    train_path: Union[Path, str, None] = None,
    eval_path: Union[Path, str, None] = None,
    model_output: Union[Path, str, None] = None,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    random_state: int = settings.random_state,
    experiment_name: str = settings.mlflow_experiment_name,
    model_registry_name: str = settings.mlflow_model_name,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run Optuna hyperparameter tuning, log trials to MLflow, and register best model."""
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

    print(f"🎯 Starting Optuna study ({n_trials} trials) with MLflow tracking...")

    parent_run_name = f"optuna_study_{n_trials}_trials"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tag("pipeline", "tuning")
        mlflow.log_param("n_trials", n_trials)
        if sample_frac:
            mlflow.log_param("sample_frac", sample_frac)

        def objective(trial: optuna.Trial) -> float:
            params = {
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

            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                model = XGBRegressor(**params)
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
        print(f"🏆 Best trial #{study.best_trial.number} RMSE: {study.best_value:.2f}s")
        print(f"   Best params: {best_params}")

        # Retrain best model
        print("🔨 Retraining best model with optimal parameters...")
        final_params = {
            **best_params,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        best_model = XGBRegressor(**final_params)
        best_model.fit(X_train, y_train)

        # Evaluate best model
        y_pred = best_model.predict(X_eval)
        best_metrics = {
            "rmse": float(root_mean_squared_error(y_eval, y_pred)),
            "mae": float(mean_absolute_error(y_eval, y_pred)),
            "r2": float(r2_score(y_eval, y_pred)),
        }

        # Log best metrics to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(best_metrics)

        # Log and Register model in MLflow Model Registry
        print(f"📦 Logging model artifact and registering in MLflow Model Registry ({model_registry_name})...")
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            registered_model_name=model_registry_name,
        )

        # Set model version alias/description
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
            print(f"🏷️ Tagged {model_registry_name} version {current_ver} as Production")

    # Persist locally to models/best_model.pkl
    dump(best_model, out_path)
    print(f"💾 Saved best model locally to {out_path}")
    print(f"📊 Best Tuned Model Performance:")
    print(f"   RMSE: {best_metrics['rmse']:.2f} seconds")
    print(f"   MAE:  {best_metrics['mae']:.2f} seconds")
    print(f"   R²:   {best_metrics['r2']:.4f}")

    return best_params, best_metrics


if __name__ == "__main__":
    tune_model(n_trials=10)
