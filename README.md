# NYC Taxi Trip Duration — Production MLOps System

An end-to-end, modular Machine Learning system for predicting NYC Yellow Taxi trip durations. Refactored from exploratory Jupyter notebooks into decoupled pipelines with Optuna, MLflow Model Registry, FastAPI, Streamlit, and Docker.

---

## Architecture

```
                       Raw Dataset (data/NYC.csv)
                                  │
                       ┌──────────▼──────────┐
                       │   load_and_split    │ (Time-Aware Split: Jan-Apr, May, June)
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │     preprocess      │ (Outlier & Coordinate Filtering)
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │ feature_engineering │ (Vectorized Haversine + Scaling & One-Hot)
                       └─────┬─────────┬─────┘
                             │         │
                   ┌─────────▼─┐     ┌─▼─────────┐
                   │  models/  │     │ processed │
                   │  .pkl     │     │ CSVs      │
                   └─────┬─────┘     └─┬─────────┘
                         │             │
        ┌────────────────┼─────────────┼───────────────┐
        │                │             │               │
  ┌─────▼──────┐  ┌──────▼──────┐  ┌───▼────────┐  ┌───▼────────┐
  │  train.py  │  │   tune.py   │  │  eval.py   │  │inference.py│
  │ (Baseline) │  │ (Optuna +   │  │ (Holdout)  │  │ (Skew-Free)│
  └────────────┘  │  MLflow)    │  └────────────┘  └───┬────────┘
                  └─────────────┘                      │
                                               ┌───────┴───────┐
                                               │               │
                                        ┌──────▼─────┐  ┌──────▼──────┐
                                        │  FastAPI   │  │  Streamlit  │
                                        │  REST API  │  │  Dashboard  │
                                        └────────────┘  └─────────────┘
```

---

## Project Structure

```
nyc-trip/
├── src/
│   ├── config.py                 # Centralized Pydantic settings and thresholds
│   ├── feature_pipeline/         # Data loading, cleaning, and feature engineering
│   │   ├── load.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── training_pipeline/        # Model training and optimization
│   │   ├── train.py              # Baseline training (LightGBM & XGBoost)
│   │   ├── tune.py               # Optuna Bayesian tuning + MLflow Model Registry
│   │   └── eval.py               # Holdout evaluation report
│   ├── inference_pipeline/       # Decoupled inference engine
│   │   └── inference.py          # predict() with strict schema alignment
│   ├── batch/                    # Batch processing runners
│   │   └── run_batch.py
│   └── api/                      # FastAPI service with Pydantic validation
│       └── main.py
├── app.py                        # Interactive Streamlit dashboard
├── models/                       # Serialized preprocessor (.pkl) and schema (.json)
├── tests/                        # 16 unit, integration, and API tests
├── Dockerfile                    # API container image
├── Dockerfile.streamlit          # Streamlit dashboard container image
├── pytest.ini                    # Pytest configuration
├── pyproject.toml                # Project metadata and dependencies
└── requirements.txt              # Production dependencies
```

---

## Quickstart and Pipeline Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Feature Pipeline
```bash
# 1. Load and time-split raw data
python -m src.feature_pipeline.load

# 2. Clean data and filter outliers
python -m src.feature_pipeline.preprocess

# 3. Compute vectorized features and save preprocessor.pkl
python -m src.feature_pipeline.feature_engineering
```

### 3. Model Training and Optuna Tuning
```bash
# Train baseline model (LightGBM or XGBoost)
python -m src.training_pipeline.train --model_type lightgbm

# Run Bayesian hyperparameter tuning with Optuna and MLflow
python -m src.training_pipeline.tune --model_type lightgbm --n_trials 10

# Evaluate best model on unseen June holdout data
python -m src.training_pipeline.eval
```

### 4. Explore Experiments in MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Open http://127.0.0.1:5000 to inspect trial metrics and the registered nyc_taxi_trip_duration_model.

---

## Serving and Dashboards

### FastAPI REST Service
```bash
uvicorn src.api.main:app --reload --port 8000
```
Interactive Swagger API documentation: http://127.0.0.1:8000/docs.

Example Single Trip Prediction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pickup_datetime": "2016-06-15 14:30:00",
       "vendor_id": 1,
       "pickup_longitude": -73.9855,
       "pickup_latitude": 40.7580,
       "dropoff_longitude": -73.9772,
       "dropoff_latitude": 40.7527
     }'
```

### Interactive Streamlit Dashboard
Live Demo: https://nyc-trip-duration-prediction.streamlit.app/

Local execution:
```bash
streamlit run app.py
```
Open http://localhost:8501 to test trips with landmark presets, view map routes, and inspect holdout metrics.

---

## Testing Suite

Run the full automated test suite:
```bash
pytest -v
```
All 16 tests cover:
* Coordinate distance formulas and date feature extraction
* Data leakage prevention and coordinate outlier filtering
* Skew-free inference smoke tests and scrambled column ordering
* Pydantic schema validation rejecting invalid GPS coordinates
* Fast training and holdout evaluation output integrity

---

## Docker Deployment

Build and run API container:
```bash
docker build -t nyc-taxi-api .
docker run -p 8000:8000 nyc-taxi-api
```

Build and run Streamlit dashboard container:
```bash
docker build -t nyc-taxi-dashboard -f Dockerfile.streamlit .
docker run -p 8501:8501 nyc-taxi-dashboard
```
