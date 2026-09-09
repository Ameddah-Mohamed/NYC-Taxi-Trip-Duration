"""
FastAPI Service with Pydantic Validation for NYC Taxi Trip Duration.

- Exposes /health, /predict, /predict/batch, and /latest_predictions.
- Enforces geographic bounding box and vendor ID constraints using Pydantic.
- Calculates estimated distance and duration in both seconds and minutes.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.config import settings
from src.inference_pipeline.inference import predict
from src.feature_pipeline.feature_engineering import haversine_distance

app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    description="High-performance machine learning inference service for NYC yellow taxi duration forecasting.",
    version="1.0.0",
)


# -------------------------------------------------------------
# Pydantic Schemas for Strict Input & Output Validation
# -------------------------------------------------------------
class TripRequest(BaseModel):
    pickup_datetime: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Trip pickup datetime string (YYYY-MM-DD HH:MM:SS)",
        examples=["2016-06-15 14:30:00"],
    )
    vendor_id: int = Field(
        ...,
        ge=1,
        le=2,
        description="Taxi technology provider (1 = Creative Mobile, 2 = VeriFone Inc)",
        examples=[1],
    )
    pickup_longitude: float = Field(
        ...,
        ge=settings.min_longitude,
        le=settings.max_longitude,
        description="Valid NYC pickup longitude (-74.50 to -73.00)",
        examples=[-73.9855],
    )
    pickup_latitude: float = Field(
        ...,
        ge=settings.min_latitude,
        le=settings.max_latitude,
        description="Valid NYC pickup latitude (40.50 to 41.00)",
        examples=[40.7580],
    )
    dropoff_longitude: float = Field(
        ...,
        ge=settings.min_longitude,
        le=settings.max_longitude,
        description="Valid NYC dropoff longitude (-74.50 to -73.00)",
        examples=[-73.9772],
    )
    dropoff_latitude: float = Field(
        ...,
        ge=settings.min_latitude,
        le=settings.max_latitude,
        description="Valid NYC dropoff latitude (40.50 to 41.00)",
        examples=[40.7527],
    )

    @model_validator(mode="after")
    def check_distinct_locations(self) -> "TripRequest":
        dist = haversine_distance(
            self.pickup_longitude,
            self.pickup_latitude,
            self.dropoff_longitude,
            self.dropoff_latitude,
        )
        if dist < 0.05:
            raise ValueError("Pickup and dropoff locations cannot be identical (distance must be at least 50 meters).")
        return self


class TripResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    estimated_distance_km: float


class BatchTripRequest(BaseModel):
    trips: List[TripRequest]


class BatchTripResponse(BaseModel):
    count: int
    predictions: List[TripResponse]


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "service": "NYC Taxi Trip Duration API",
        "status": "online",
        "docs": "/docs",
        "model_file": settings.best_model_path.name,
    }


@app.get("/health", tags=["General"])
def health_check():
    model_exists = settings.best_model_path.exists() or settings.baseline_model_path.exists()
    prep_exists = settings.preprocessor_path.exists()

    is_healthy = model_exists and prep_exists
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model_exists,
        "preprocessor_loaded": prep_exists,
        "active_model": str(settings.best_model_path if settings.best_model_path.exists() else settings.baseline_model_path),
    }


@app.post("/predict", response_model=TripResponse, tags=["Inference"])
def predict_single_trip(trip: TripRequest):
    """Predict duration for a single taxi trip."""
    try:
        raw_dict = trip.model_dump()
        result_df = predict(raw_dict)

        dist = float(
            haversine_distance(
                trip.pickup_longitude,
                trip.pickup_latitude,
                trip.dropoff_longitude,
                trip.dropoff_latitude,
            )
        )

        pred_sec = float(result_df["predicted_duration_seconds"].iloc[0])
        pred_min = float(result_df["predicted_duration_minutes"].iloc[0])

        return TripResponse(
            predicted_duration_seconds=round(pred_sec, 1),
            predicted_duration_minutes=round(pred_min, 2),
            estimated_distance_km=round(dist, 2),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchTripResponse, tags=["Inference"])
def predict_batch_trips(payload: BatchTripRequest):
    """Predict duration for a batch of taxi trips."""
    if not payload.trips:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Trips list cannot be empty")

    try:
        raw_list = [t.model_dump() for t in payload.trips]
        result_df = predict(raw_list)

        predictions: List[TripResponse] = []
        for i, row in result_df.iterrows():
            t = payload.trips[i]
            dist = float(
                haversine_distance(
                    t.pickup_longitude,
                    t.pickup_latitude,
                    t.dropoff_longitude,
                    t.dropoff_latitude,
                )
            )
            predictions.append(
                TripResponse(
                    predicted_duration_seconds=float(row["predicted_duration_seconds"]),
                    predicted_duration_minutes=float(row["predicted_duration_minutes"]),
                    estimated_distance_km=round(dist, 2),
                )
            )

        return BatchTripResponse(count=len(predictions), predictions=predictions)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )


@app.get("/latest_predictions", tags=["Batch"])
def get_latest_predictions(limit: int = 5):
    """Preview recent batch prediction results from disk."""
    pred_files = sorted(settings.predictions_dir.glob("*.csv"))
    if not pred_files:
        return {"message": "No batch predictions available yet. Run a batch job first."}

    latest = pred_files[-1]
    df = pd.read_csv(latest)
    return {
        "file": latest.name,
        "total_rows": len(df),
        "preview": df.head(limit).to_dict(orient="records"),
    }
