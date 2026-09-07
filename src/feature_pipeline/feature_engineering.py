"""
Feature Engineering Module for NYC Taxi Trip Duration Pipeline.

- Vectorized distance computation (distance_km via Haversine).
- Temporal feature extraction (pickup_hour, pickup_dayofweek, pickup_month).
- Fits ColumnTransformer (StandardScaler + OneHotEncoder) strictly on train.
- Persists preprocessor to models/preprocessor.pkl and schema to models/feature_columns.json.
- Transforms and exports engineered train, eval, and holdout datasets.
"""

import json
from pathlib import Path
from typing import List, Tuple, Union
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings


def haversine_distance(
    lon1: Union[np.ndarray, pd.Series, float],
    lat1: Union[np.ndarray, pd.Series, float],
    lon2: Union[np.ndarray, pd.Series, float],
    lat2: Union[np.ndarray, pd.Series, float],
) -> np.ndarray:
    """Vectorized Haversine great-circle distance in kilometers between coordinate pairs.
    
    Vectorized NumPy is ~100x faster than row-by-row geodesic calculations,
    with less than 0.1 km difference across NYC metropolitan coordinates.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal components from pickup_datetime."""
    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"])
        df["pickup_hour"] = dt.dt.hour
        df["pickup_dayofweek"] = dt.dt.dayofweek
        df["pickup_month"] = dt.dt.month
        df = df.drop(columns=["pickup_datetime"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the DataFrame."""
    df = df.copy()

    # Temporal features
    df = extract_datetime_features(df)

    # Vectorized distance feature
    coord_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    if all(c in df.columns for c in coord_cols):
        df["distance_km"] = haversine_distance(
            df["pickup_longitude"].values,
            df["pickup_latitude"].values,
            df["dropoff_longitude"].values,
            df["dropoff_latitude"].values,
        )

    return df


def build_preprocessor(
    num_cols: List[str] = settings.num_columns,
    cat_cols: List[str] = settings.cat_columns,
) -> ColumnTransformer:
    """Construct ColumnTransformer for numerical scaling and categorical encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def run_feature_engineering(
    in_train_path: Union[Path, str, None] = None,
    in_eval_path: Union[Path, str, None] = None,
    in_holdout_path: Union[Path, str, None] = None,
    output_dir: Union[Path, str, None] = None,
    models_dir: Union[Path, str, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Execute full feature engineering and persistence pipeline."""
    proc_dir = Path(output_dir) if output_dir is not None else settings.processed_dir
    mod_dir = Path(models_dir) if models_dir is not None else settings.models_dir
    proc_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    train_file = Path(in_train_path) if in_train_path else proc_dir / "cleaned_train.csv"
    eval_file = Path(in_eval_path) if in_eval_path else proc_dir / "cleaned_eval.csv"
    holdout_file = Path(in_holdout_path) if in_holdout_path else proc_dir / "cleaned_holdout.csv"

    print("⚙Loading cleaned splits...")
    train_df = pd.read_csv(train_file)
    eval_df = pd.read_csv(eval_file)
    holdout_df = pd.read_csv(holdout_file)

    # 1. Add engineered features
    train_df = add_features(train_df)
    eval_df = add_features(eval_df)
    holdout_df = add_features(holdout_df)

    # 2. Filter stationary/zero distance trips
    train_df = train_df[train_df["distance_km"] > settings.min_distance_km].reset_index(drop=True)
    eval_df = eval_df[eval_df["distance_km"] > settings.min_distance_km].reset_index(drop=True)
    holdout_df = holdout_df[holdout_df["distance_km"] > settings.min_distance_km].reset_index(drop=True)

    # Separate targets
    target = "trip_duration"
    y_train = train_df[target]
    y_eval = eval_df[target]
    y_holdout = holdout_df[target] if target in holdout_df.columns else None

    X_train_raw = train_df[settings.num_columns + settings.cat_columns]
    X_eval_raw = eval_df[settings.num_columns + settings.cat_columns]
    X_holdout_raw = holdout_df[settings.num_columns + settings.cat_columns]

    # 3. Fit preprocessor on training data only
    print("Fitting preprocessor on training split...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_eval_transformed = preprocessor.transform(X_eval_raw)
    X_holdout_transformed = preprocessor.transform(X_holdout_raw)

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Preprocessing fitted. Generated {len(feature_names)} features: {feature_names}")

    # Build final transformed DataFrames
    df_train_fe = pd.DataFrame(X_train_transformed, columns=feature_names)
    df_train_fe[target] = y_train.values

    df_eval_fe = pd.DataFrame(X_eval_transformed, columns=feature_names)
    df_eval_fe[target] = y_eval.values

    df_holdout_fe = pd.DataFrame(X_holdout_transformed, columns=feature_names)
    if y_holdout is not None:
        df_holdout_fe[target] = y_holdout.values

    # 4. Save artifacts
    preprocessor_path = mod_dir / "preprocessor.pkl"
    dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

    feature_cols_path = mod_dir / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved feature column schema to {feature_cols_path}")

    # 5. Save engineered datasets
    out_train_path = proc_dir / "feature_engineered_train.csv"
    out_eval_path = proc_dir / "feature_engineered_eval.csv"
    out_holdout_path = proc_dir / "feature_engineered_holdout.csv"

    df_train_fe.to_csv(out_train_path, index=False)
    df_eval_fe.to_csv(out_eval_path, index=False)
    df_holdout_fe.to_csv(out_holdout_path, index=False)

    print(f"Feature engineering complete:")
    print(f"   Train:   {df_train_fe.shape} -> {out_train_path}")
    print(f"   Eval:    {df_eval_fe.shape} -> {out_eval_path}")
    print(f"   Holdout: {df_holdout_fe.shape} -> {out_holdout_path}")

    return df_train_fe, df_eval_fe, df_holdout_fe, preprocessor


if __name__ == "__main__":
    run_feature_engineering()
