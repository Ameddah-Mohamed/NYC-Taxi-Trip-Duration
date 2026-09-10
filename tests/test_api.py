"""Tests for FastAPI service endpoints and Pydantic validation."""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_trip_success():
    payload = {
        "pickup_datetime": "2016-05-20 12:00:00",
        "vendor_id": 1,
        "pickup_longitude": -73.9855,
        "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.9772,
        "dropoff_latitude": 40.7527,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_duration_seconds" in data
    assert "predicted_duration_minutes" in data
    assert "estimated_distance_km" in data
    assert data["predicted_duration_seconds"] > 0
    assert data["estimated_distance_km"] > 0


def test_predict_invalid_coordinates_rejected():
    # Latitude 99.0 is way outside NYC and invalid
    bad_payload = {
        "pickup_datetime": "2016-05-20 12:00:00",
        "vendor_id": 1,
        "pickup_longitude": -73.9855,
        "pickup_latitude": 99.0,
        "dropoff_longitude": -73.9772,
        "dropoff_latitude": 40.7527,
    }
    response = client.post("/predict", json=bad_payload)
    # Pydantic validation error returns 422 Unprocessable Entity
    assert response.status_code == 422


def test_predict_invalid_vendor_rejected():
    # Vendor must be 1 or 2
    bad_payload = {
        "pickup_datetime": "2016-05-20 12:00:00",
        "vendor_id": 99,
        "pickup_longitude": -73.9855,
        "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.9772,
        "dropoff_latitude": 40.7527,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_identical_locations_rejected():
    # Identical pickup and dropoff must be rejected
    bad_payload = {
        "pickup_datetime": "2016-05-20 12:00:00",
        "vendor_id": 1,
        "pickup_longitude": -73.9772,
        "pickup_latitude": 40.7663,
        "dropoff_longitude": -73.9772,
        "dropoff_latitude": 40.7663,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "distinct" in response.text.lower() or "identical" in response.text.lower()


def _valid_trip(vendor_id=1, pickup_lon=-73.9855):
    return {
        "pickup_datetime": "2016-05-20 12:00:00",
        "vendor_id": vendor_id,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.9772,
        "dropoff_latitude": 40.7527,
    }


def test_predict_batch_success():
    payload = {"trips": [_valid_trip(), _valid_trip(vendor_id=2, pickup_lon=-73.9900)]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["predicted_duration_seconds"] >= 100.0
        assert pred["predicted_duration_minutes"] > 0
        assert pred["estimated_distance_km"] > 0


def test_predict_batch_empty_rejected():
    response = client.post("/predict/batch", json={"trips": []})
    assert response.status_code == 400


def test_predict_batch_invalid_trip_rejected():
    payload = {"trips": [_valid_trip(), {**_valid_trip(), "vendor_id": 99}]}
    response = client.post("/predict/batch", json=payload)
    # Per-trip Pydantic validation fails the whole batch
    assert response.status_code == 422

