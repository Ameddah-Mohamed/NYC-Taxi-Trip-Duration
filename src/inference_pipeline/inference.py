"""
Production Inference Pipeline for NYC Taxi Trip Duration Prediction.

- Ingests raw input data (single dictionary, list of dicts, or raw DataFrame).
- Applies identical preprocessing and vectorized feature engineering.
- Transforms features using persisted preprocessor (models/preprocessor.pkl).
- Enforces strict schema alignment against training columns (models/feature_columns.json).
- Returns duration predictions (seconds and minutes).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from joblib import load
import numpy as np
import pandas as pd

from src.config import settings
from src.feature_pipeline.feature_engineering import add_features
from src.feature_pipeline.preprocess import drop_leakage_and_metadata


def load_inference_artifacts(
    model_path: Union[Path, str, None] = None,
    preprocessor_path: Union[Path, str, None] = None,
    feature_columns_path: Union[Path, str, None] = None,
) -> Tuple[Any, Any, List[str]]:
    """Load the serialized model, preprocessor, and schema list."""
    m_path = Path(model_path) if model_path is not None else settings.best_model_path
    p_path = Path(preprocessor_path) if preprocessor_path is not None else settings.preprocessor_path
    f_path = Path(feature_columns_path) if feature_columns_path is not None else settings.feature_columns_path

    if not m_path.exists():
        # Fallback to baseline if best_model not tuned yet
        m_path = settings.baseline_model_path
    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {m_path}")

    if not p_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {p_path}")

    model = load(m_path)
    preprocessor = load(p_path)

    if f_path.exists():
        with open(f_path, "r") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = list(preprocessor.get_feature_names_out())

    return model, preprocessor, feature_cols


def predict(
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: Union[Path, str, None] = None,
    preprocessor_path: Union[Path, str, None] = None,
    feature_columns_path: Union[Path, str, None] = None,
) -> pd.DataFrame:
    """Run end-to-end inference on raw input records.

    Parameters
    ----------
    input_data : pd.DataFrame | dict | list[dict]
        Raw input trip data containing pickup_datetime, vendor_id, and coordinates.

    Returns
    -------
    pd.DataFrame
        DataFrame with original inputs, predicted_duration_seconds, and predicted_duration_minutes.
    """
    # 1. Normalize input to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    if df.empty:
        return pd.DataFrame(columns=["predicted_duration_seconds", "predicted_duration_minutes"])

    original_df = df.copy()

    # 2. Preserve ground truth if present in input
    y_actual = None
    if "trip_duration" in df.columns:
        y_actual = df["trip_duration"].values
        df = df.drop(columns=["trip_duration"])

    # 3. Clean and engineer features
    df = drop_leakage_and_metadata(df)
    df = add_features(df)

    # 4. Load artifacts
    model, preprocessor, expected_feature_cols = load_inference_artifacts(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        feature_columns_path=feature_columns_path,
    )

    # Ensure required input columns for the preprocessor are present
    required_raw_cols = settings.num_columns + settings.cat_columns
    missing_cols = [c for c in required_raw_cols if c not in df.columns]
    for col in missing_cols:
        df[col] = 0.0

    X_raw = df[required_raw_cols]

    # 5. Transform features using the training preprocessor
    X_transformed = preprocessor.transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)

    # 6. Strict Schema Alignment (matches training columns exactly)
    df_aligned = df_transformed.reindex(columns=expected_feature_cols, fill_value=0.0)

    # 7. Generate predictions
    raw_preds = model.predict(df_aligned)
    # Floor predictions at min duration threshold (prevent negative predictions)
    predictions_sec = np.maximum(raw_preds, settings.min_trip_duration)

    # 8. Assemble result
    results_df = original_df.copy()
    results_df["predicted_duration_seconds"] = np.round(predictions_sec, 1)
    results_df["predicted_duration_minutes"] = np.round(predictions_sec / 60.0, 2)

    if y_actual is not None:
        results_df["actual_duration_seconds"] = y_actual
        results_df["actual_duration_minutes"] = np.round(y_actual / 60.0, 2)
        results_df["absolute_error_seconds"] = np.round(np.abs(results_df["predicted_duration_seconds"] - y_actual), 1)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on raw NYC taxi data.")
    parser.add_argument("--input", type=str, required=True, help="Path to input raw CSV file")
    parser.add_argument("--output", type=str, default="data/predictions/predictions.csv", help="Path to save output CSV")
    parser.add_argument("--model", type=str, default=None, help="Optional model path")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    preds = predict(input_df, model_path=args.model)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_file, index=False)
    print(f"✅ Generated predictions for {len(preds)} trips -> saved to {out_file}")
    print(preds[["predicted_duration_seconds", "predicted_duration_minutes"]].head())
