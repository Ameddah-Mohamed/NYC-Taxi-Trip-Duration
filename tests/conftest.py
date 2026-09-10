"""Pytest configuration and global fixtures for CI & local test runs."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from joblib import dump, load
from lightgbm import LGBMRegressor

from src.config import settings
from src.feature_pipeline.feature_engineering import build_preprocessor


@pytest.fixture(scope="session", autouse=True)
def setup_ci_test_environment():
    """Ensure minimal models and schemas exist so tests pass in clean CI environments without raw data."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)

    # Cyclic time features (unit-circle values) + flags-: 13 num + vendor cat = 15 transformed features
    sample_data = {
        "vendor_id": [1, 2, 1, 2],
        "pickup_longitude": [-73.98, -73.97, -73.99, -73.96],
        "pickup_latitude": [40.75, 40.76, 40.74, 40.77],
        "dropoff_longitude": [-73.97, -73.98, -73.96, -73.95],
        "dropoff_latitude": [40.76, 40.75, 40.77, 40.74],
        "distance_km": [1.2, 2.5, 3.1, 0.8],
        "pickup_hour_sin": [0.87, 0.0, -1.0, -0.5],
        "pickup_hour_cos": [-0.5, -1.0, 0.0, 0.87],
        "pickup_dow_sin": [0.0, 0.97, -0.43, -0.78],
        "pickup_dow_cos": [1.0, -0.22, -0.9, 0.62],
        "pickup_month_sin": [0.0, 0.87, 1.0, 0.5],
        "pickup_month_cos": [1.0, 0.5, 0.0, -0.87],
        "is_rush_hour": [1.0, 0.0, 1.0, 0.0],
        "is_weekend": [0.0, 0.0, 0.0, 1.0],
        "trip_duration": [300, 600, 750, 200],
    }
    sample_df = pd.DataFrame(sample_data)

    # 1. Ensure preprocessor artifact exists
    if not settings.preprocessor_path.exists():
        prep = build_preprocessor()
        X_raw = sample_df[settings.num_columns + settings.cat_columns]
        X_transformed = prep.fit_transform(X_raw)
        dump(prep, settings.preprocessor_path)

        feature_cols = list(prep.get_feature_names_out())
        with open(settings.feature_columns_path, "w") as f:
            json.dump(feature_cols, f, indent=2)

    # 2. Ensure model artifacts exist
    if not settings.best_model_path.exists():
        prep = load(settings.preprocessor_path)
        feature_cols = list(prep.get_feature_names_out())
        X_dummy = np.zeros((10, len(feature_cols)))
        y_dummy = np.array([400.0] * 10)

        dummy_model = LGBMRegressor(n_estimators=5, random_state=14, verbose=-1)
        dummy_model.fit(X_dummy, y_dummy)
        dump(dummy_model, settings.best_model_path)

    if not settings.baseline_model_path.exists():
        from shutil import copyfile
        copyfile(settings.best_model_path, settings.baseline_model_path)
