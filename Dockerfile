# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-gcp.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-gcp.txt

# Copy application code
COPY app.py .
COPY preprocessing.py .
COPY templates/ ./templates/

# Copy model artifacts
COPY best_model.h5 .
COPY scaler.pkl .
COPY X_test.npy .
COPY y_test.npy .
COPY lstm_y_pred.npy .
COPY gru_y_pred.npy .
COPY y_true.npy .
COPY dataset/ ./dataset/

# Allow statements and log messages to immediately appear
ENV PYTHONUNBUFFERED True

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
