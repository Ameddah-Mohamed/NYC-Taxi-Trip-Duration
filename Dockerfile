# Lightweight Python base image
FROM python:3.11-slim

WORKDIR /app

# Install OpenMP runtime for LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
