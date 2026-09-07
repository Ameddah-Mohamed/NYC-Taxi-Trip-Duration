"""Unit tests for feature pipeline and data transformations."""

import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline.feature_engineering import add_features, extract_datetime_features, haversine_distance
from src.feature_pipeline.preprocess import clean_coordinates, drop_leakage_and_metadata, filter_duration_outliers


def test_haversine_distance_known_coordinates():
    # Times Square (40.7580, -73.9855) to Grand Central (40.7527, -73.9772)
    # Approximate straight-line distance is ~0.9 km
    dist = haversine_distance(-73.9855, 40.7580, -73.9772, 40.7527)
    assert 0.8 < dist < 1.0


def test_extract_datetime_features():
    df = pd.DataFrame({
        "pickup_datetime": ["2016-03-15 17:45:00", "2016-06-01 02:10:00"]
    })
    df_feat = extract_datetime_features(df)
    assert "pickup_hour" in df_feat.columns
    assert "pickup_dayofweek" in df_feat.columns
    assert "pickup_month" in df_feat.columns
    assert df_feat["pickup_hour"].tolist() == [17, 2]
    assert df_feat["pickup_month"].tolist() == [3, 6]
    assert "pickup_datetime" not in df_feat.columns


def test_drop_leakage_columns():
    df = pd.DataFrame({
        "id": ["id123"],
        "dropoff_datetime": ["2016-03-15 18:00:00"],
        "vendor_id": [1],
        "trip_duration": [500],
    })
    cleaned = drop_leakage_and_metadata(df)
    assert "id" not in cleaned.columns
    assert "dropoff_datetime" not in cleaned.columns
    assert "vendor_id" in cleaned.columns


def test_filter_duration_outliers():
    df = pd.DataFrame({
        "trip_duration": [50, 100, 500, 4000, 4001, 10000]
    })
    filtered = filter_duration_outliers(df, min_duration=100, max_duration=4000)
    assert set(filtered["trip_duration"].tolist()) == {500, 4000}


def test_clean_coordinates_outliers():
    df = pd.DataFrame({
        "pickup_latitude": [40.75, 50.0],       # 50.0 is outside NYC
        "pickup_longitude": [-73.98, -73.98],
        "dropoff_latitude": [40.76, 40.76],
        "dropoff_longitude": [-73.97, -73.97],
    })
    cleaned = clean_coordinates(df)
    assert len(cleaned) == 1
    assert cleaned["pickup_latitude"].iloc[0] == 40.75
