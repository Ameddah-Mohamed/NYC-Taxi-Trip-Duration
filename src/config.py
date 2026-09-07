"""
Centralized configuration for NYC Taxi Trip Duration MLOps pipeline.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Project paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "NYC.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    predictions_dir: Path = PROJECT_ROOT / "data" / "predictions"
    models_dir: Path = PROJECT_ROOT / "models"

    # Processed data splits
    train_path: Path = processed_dir / "train.csv"
    eval_path: Path = processed_dir / "eval.csv"
    holdout_path: Path = processed_dir / "holdout.csv"

    # Persisted model & transformer artifacts
    best_model_path: Path = models_dir / "best_model.pkl"
    baseline_model_path: Path = models_dir / "baseline_model.pkl"
    preprocessor_path: Path = models_dir / "preprocessor.pkl"
    feature_columns_path: Path = models_dir / "feature_columns.json"

    # Data filtering & cleaning thresholds
    min_trip_duration: int = 100
    max_trip_duration: int = 4000
    min_distance_km: float = 0.1

    # NYC Geographic bounding box
    min_latitude: float = 40.50
    max_latitude: float = 41.00
    min_longitude: float = -74.50
    max_longitude: float = -73.00

    # Categorical and numerical columns
    cat_columns: List[str] = Field(default_factory=lambda: ["vendor_id", "pickup_month"])
    num_columns: List[str] = Field(
        default_factory=lambda: [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_hour",
            "pickup_dayofweek",
            "distance_km",
        ]
    )

    # MLflow & Experiment Tracking
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment_name: str = "nyc_taxi_duration_prediction"
    mlflow_model_name: str = "nyc_taxi_trip_duration_model"

    # Reproducibility
    random_state: int = 14
    test_size: float = 0.2
    eval_size: float = 0.1

    # API & Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Global singleton instance
settings = Settings()
