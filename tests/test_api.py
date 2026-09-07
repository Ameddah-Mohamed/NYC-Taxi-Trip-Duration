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
