"""Smoke and integration tests for the inference pipeline."""

import pandas as pd
import pytest

from src.inference_pipeline.inference import predict


@pytest.fixture
def sample_raw_trip():
    return {
        "vendor_id": 1,
        "pickup_datetime": "2016-04-12 18:30:00",
        "pickup_longitude": -73.9855,
        "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.9772,
        "dropoff_latitude": 40.7527,
    }


def test_inference_single_sample(sample_raw_trip):
    res = predict(sample_raw_trip)
    assert not res.empty
    assert "predicted_duration_seconds" in res.columns
    assert "predicted_duration_minutes" in res.columns
    pred_sec = res["predicted_duration_seconds"].iloc[0]
    assert pred_sec >= 100.0  # Must obey physical floor


def test_inference_scrambled_column_order(sample_raw_trip):
    # Reorder keys randomly
    scrambled = {
        "dropoff_latitude": sample_raw_trip["dropoff_latitude"],
        "vendor_id": sample_raw_trip["vendor_id"],
        "pickup_longitude": sample_raw_trip["pickup_longitude"],
        "pickup_datetime": sample_raw_trip["pickup_datetime"],
        "pickup_latitude": sample_raw_trip["pickup_latitude"],
        "dropoff_longitude": sample_raw_trip["dropoff_longitude"],
    }
    res = predict(scrambled)
    assert not res.empty
    assert res["predicted_duration_seconds"].iloc[0] > 0


def test_inference_extra_unwanted_columns(sample_raw_trip):
    with_extra = {
        **sample_raw_trip,
        "random_extra_col": "testing_metadata",
        "id": "taxi_9999",
        "passenger_count": 4,
    }
    res = predict(with_extra)
    assert not res.empty
    assert "predicted_duration_seconds" in res.columns


def test_inference_batch(sample_raw_trip):
    trips = [sample_raw_trip, sample_raw_trip]
    df = pd.DataFrame(trips)
    res = predict(df)
    assert len(res) == 2
    assert "predicted_duration_seconds" in res.columns
