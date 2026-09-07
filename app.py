"""
Interactive Streamlit Dashboard for NYC Taxi Trip Duration Forecasting.

Features:
- Real-time trip duration estimation with preset NYC landmarks
- Interactive map plotting pickup and dropoff points
- Analytics panel displaying holdout evaluation metrics
- Direct integration with the inference pipeline or FastAPI
"""

from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import settings
from src.feature_pipeline.feature_engineering import haversine_distance
from src.inference_pipeline.inference import predict

st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="",
    layout="wide",
)

# Preset NYC Landmarks for quick testing
NYC_LANDMARKS = {
    "Times Square": (40.7580, -73.9855),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Empire State Building": (40.7484, -73.9857),
    "Central Park South": (40.7660, -73.9772),
    "Wall Street / Financial District": (40.7075, -74.0090),
    "Brooklyn Bridge (Manhattan side)": (40.7100, -74.0000),
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport (LGA)": (40.7769, -73.8740),
}

st.title("New York City Taxi Trip Duration Predictor")
st.markdown(
    "Production-grade Machine Learning inference engine for estimating NYC Yellow Taxi travel times."
)

tab1, tab2, tab3 = st.tabs(["Live Trip Estimator", "Model Performance & Metrics", "Batch Prediction Explorer"])

# -------------------------------------------------------------
# TAB 1: Live Trip Estimator
# -------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Configure Trip")

        use_presets = st.checkbox("Select from popular NYC Landmarks", value=True)

        if use_presets:
            col_p, col_d = st.columns(2)
            with col_p:
                pickup_name = st.selectbox("Pickup Location", list(NYC_LANDMARKS.keys()), index=0)
                pickup_lat, pickup_lon = NYC_LANDMARKS[pickup_name]
            with col_d:
                dropoff_name = st.selectbox("Dropoff Location", list(NYC_LANDMARKS.keys()), index=1)
                dropoff_lat, dropoff_lon = NYC_LANDMARKS[dropoff_name]
        else:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pickup_lat = st.number_input("Pickup Latitude", value=40.7580, format="%.5f")
                pickup_lon = st.number_input("Pickup Longitude", value=-73.9855, format="%.5f")
            with col_p2:
                dropoff_lat = st.number_input("Dropoff Latitude", value=40.7527, format="%.5f")
                dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9772, format="%.5f")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            trip_date = st.date_input("Pickup Date", value=datetime(2016, 6, 15))
        with col_t2:
            trip_time = st.time_input("Pickup Time", value=datetime.now().time())

        vendor = st.radio("Taxi Technology Provider (Vendor ID)", [1, 2], index=0, horizontal=True)

        combined_datetime = f"{trip_date} {trip_time.strftime('%H:%M:%S')}"

        predict_btn = st.button("Calculate Estimated Trip Duration", use_container_width=True, type="primary")

    with col_right:
        st.subheader("Trip Route & Prediction")

        # Compute straight-line distance
        dist_km = haversine_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)

        if predict_btn or "last_pred" in st.session_state:
            if predict_btn:
                payload = {
                    "vendor_id": int(vendor),
                    "pickup_datetime": combined_datetime,
                    "pickup_longitude": float(pickup_lon),
                    "pickup_latitude": float(pickup_lat),
                    "dropoff_longitude": float(dropoff_lon),
                    "dropoff_latitude": float(dropoff_lat),
                }
                res = predict(payload)
                st.session_state["last_pred"] = {
                    "sec": float(res["predicted_duration_seconds"].iloc[0]),
                    "min": float(res["predicted_duration_minutes"].iloc[0]),
                    "dist": float(dist_km),
                }

            last = st.session_state["last_pred"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Time", f"{last['min']:.1f} min", f"{last['sec']:.0f} sec")
            m2.metric("Distance", f"{last['dist']:.2f} km", f"{last['dist'] * 0.621371:.2f} miles")
            avg_speed = (last["dist"] / (last["sec"] / 3600.0)) if last["sec"] > 0 else 0
            m3.metric("Est. Speed", f"{avg_speed:.1f} km/h")

        # Map plot
        map_df = pd.DataFrame([
            {"lat": pickup_lat, "lon": pickup_lon, "type": "Pickup "},
            {"lat": dropoff_lat, "lon": dropoff_lon, "type": "Dropoff "},
        ])
        fig = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            color="type",
            zoom=12,
            height=380,
            title="Pickup & Dropoff Locations",
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Model Performance & Metrics
# -------------------------------------------------------------
with tab2:
    st.subheader("Production Model Performance Summary")

    metrics_file = settings.models_dir / "eval_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median Absolute Error", f"{metrics.get('median_absolute_error_seconds', 0):.1f} s", "~2 mins")
        c2.metric("Mean Absolute Error (MAE)", f"{metrics.get('mae_seconds', 0):.1f} s")
        c3.metric("Root Mean Squared Error (RMSE)", f"{metrics.get('rmse_seconds', 0):.1f} s")
        c4.metric("Holdout Test Samples", f"{metrics.get('holdout_samples', 0):,}")

        st.json(metrics)
    else:
        st.info("Run `python -m src.training_pipeline.eval` to compute and inspect evaluation metrics.")

# -------------------------------------------------------------
# TAB 3: Batch Prediction Explorer
# -------------------------------------------------------------
with tab3:
    st.subheader("Latest Batch Prediction Outputs")
    pred_files = sorted(settings.predictions_dir.glob("*.csv"))
    if pred_files:
        selected_file = st.selectbox("Select Prediction Artifact", [f.name for f in pred_files])
        file_path = settings.predictions_dir / selected_file
        df_preview = pd.read_csv(file_path)
        st.write(f"Total Rows: **{len(df_preview):,}**")
        st.dataframe(df_preview.head(50), use_container_width=True)
    else:
        st.info("No batch predictions generated yet. Run `python -m src.batch.run_batch` to generate batch artifacts.")
