# SalesSense: Retail Sales Forecasting with Deep Learning

A production-ready deep learning application for predicting daily retail store sales using LSTM and GRU neural networks. Built with Flask and deployed on Google Cloud Run.

## 🎯 Objective

Build a deep learning system for retail sales forecasting that trains LSTM and GRU models on a real time series dataset and provides an interactive dashboard where users can explore trends and generate predictions.

## 🛠️ Tech Stack

- **Python** — Core language for data processing and model development
- **TensorFlow** — Deep learning framework for building and training neural networks
- **NumPy** — Numerical computing for array operations
- **Pandas** — Data manipulation and analysis
- **Scikit-learn** — Machine learning utilities (metrics, scaling)
- **Matplotlib** — Data visualization and prediction plots
- **Flask** — Web framework for API and dashboard
- **Google Cloud Run** — Cloud deployment platform

## 📋 Project Overview

This project implements a time-series forecasting system using real-world retail store inventory data (2022-2024, 731 days, 73,100 records). The goal is to predict daily sales using a 14-day sliding window with two recurrent neural network architectures: LSTM and GRU.

**Best Model:** GRU with a mean RMSE of **1,041.34 units** (5-fold cross-validation)

## ✨ Project Highlights

### 🤖 Model Training and Persistence
- Load and preprocess retail dataset (73,100 records, 731 days)
- Generate sequences with 14-day sliding window
- Train LSTM and GRU models with optimized architectures
- Evaluate performance using RMSE and MAE metrics
- Save trained models as `.h5` files for production reuse

### 🔄 Cross Validation and Model Selection
- Apply 5-fold walk-forward validation for time-series integrity
- Compare LSTM (29,345 params) vs GRU (22,305 params)
- Select GRU as best model: **1,041.34 RMSE** with 22% fewer parameters
- Validate generalization and ensure production robustness

### 🔮 Prediction Workflow
- Use last 14 days of sales data as input
- Generate next-day prediction from trained model
- Apply inverse scaling for real-world units
- Support batch predictions for multiple scenarios

### 📊 Interactive Dashboard
- Build intuitive interface with HTML/CSS/JavaScript
- Display dataset insights and performance metrics
- Allow interactive testing with custom 14-day sequences
- Real-time prediction visualization with histograms
- Responsive design for desktop and mobile

### 📈 Data Visualization
- Plot predictions vs actual values

### 🚀 Deployment
- Deploy on Google Cloud Run with automatic CI/CD
- Load trained models for real-time inference
- Container-based deployment ensures consistency
- Scalable infrastructure handles variable traffic



```
SalesSense/
├── 📂 dataset/
│   └── 📊 retail_store_inventory.csv       # Raw data (73,100 records)
├── 📂 photos/                           # Project images
├── 📂 templates/
│   └── 🌐 index.html                       # Web dashboard (HTML/CSS/JS)
├── 🐍 app.py                               # Flask web application & API
├── ⚙️ preprocessing.py                     # Data loading, normalization, sequence generation
├── 🧠 train_lstm.py                        # LSTM model training and evaluation
├── 🧠 train_gru.py                         # GRU model training and evaluation
├── 🔄 cross_validation.py                  # 5-fold walk-forward time-series CV
├── ⚖️ compare.py                           # Model comparison and metrics
├── 📈 visualise_predictions.py             # Generate prediction plots
├── 💾 best_model.h5                        # Trained GRU model (production)
├── 💾 gru_model.h5, lstm_model.h5          # Saved models
├── 🔢 X_train.npy, X_test.npy              # Preprocessed sequences
├── 🎯 y_train.npy, y_test.npy              # Target values
├── 🔮 gru_y_pred.npy, lstm_y_pred.npy      # Model predictions
├── 🐳 Dockerfile                           # Docker container configuration
├── ☁️ cloudbuild.yaml                      # Google Cloud Build config
├── 📦 requirements-gcp.txt                 # Python dependencies
├── 📖 README.md                            # This file
└── 📓 salesPredictionNotebook.ipynb        # Jupyter notebook
```

## 🚀 Quick Start

### Local Development

#### 1. Install Dependencies

```bash
pip install -r requirements-gcp.txt
```

#### 2. Preprocess Data

```bash
python preprocessing.py
```

Outputs:
- `X_train.npy`, `X_test.npy` — Normalized sequences
- `y_train.npy`, `y_test.npy` — Target values

#### 3. Train Models (Optional)

```bash
# Train LSTM
python train_lstm.py

# Train GRU
python train_gru.py

# Compare models
python compare.py

# Cross-validation evaluation
python cross_validation.py
```

#### 4. Run Flask Application

```bash
python app.py
```

Visit `http://localhost:8080` in your browser. (The app defaults to 8080).

## 🌐 Deployment on Google Cloud Run

### Prerequisites
- Google Cloud account with Billing enabled
- `gcloud` CLI installed
- Docker installed (or use Cloud Build)

### Deployment Steps

```bash
# Login to Google Cloud
gcloud auth login

# Set your GCP project
gcloud config set project YOUR_PROJECT_ID

# Deploy to Cloud Run
gcloud run deploy sales-sense \
  --source . \
  --platform managed \
  --region europe-west9 \
  --allow-unauthenticated \
  --memory 2Gi
```

After deployment, you'll receive a URL like:
```
https://sales-sense-xxxxx-ew.a.run.app
```

### Using Cloud Build (CI/CD)

The `cloudbuild.yaml` file enables automatic deployment from Git:

```bash
# Push code to trigger build
git push origin main
```

Cloud Build will:
1. Build Docker image
2. Push to Container Registry
3. Deploy to Cloud Run

## 📊 Model Architecture

Both models use the same architecture with only the recurrent layer changing:

```
Input (14, 1)
  ↓
GRU/LSTM: 50 units, dropout 0.2
  ↓
Dense: 25 units, activation=relu
  ↓
Dense: 1 unit (output)
```

**Parameters:**
- LSTM: 29,345 parameters
- GRU: 22,305 parameters (22% fewer → more efficient)

## 📈 Results

### 5-Fold Walk-Forward Cross-Validation

| Model | Mean RMSE | MAE | Performance |
|-------|-----------|-----|-------------|
| **GRU** | **1,041.34** | 854.10 | ✅ Selected |
| LSTM | 1,045.84 | 854.34 | Baseline |

**Key Findings:**
- Both models capture overall trends well
- GRU achieves lower RMSE with fewer parameters
- GRU exhibits more stable generalization across folds
- Error is ~8% relative to mean daily sales (13,200 units)

## 🖥️ Web Application Features

### Dashboard (HTML/JS)

**Prediction Section (Top):**
- Interactive 14-day sales input fields
- "Predict Day 15" button
- Real-time prediction with histogram visualization

**Analytics Section:**
- Key metrics cards (Best Model, RMSE, MAE, Test Samples)
- Predictions vs Actual chart (line plot)
- RMSE & MAE comparison (bar chart)
- Last 90 days sales trend (line chart)

## 🔌 Flask API Endpoints

### GET /
Serves the web dashboard

### GET /health
Returns health check status:
```json
{
  "status": "healthy"
}
```

### GET /api/overview
Returns model metrics:
```json
{
  "best_model": "GRU",
  "gru_cv_rmse": 1041.34,
  "gru_test_mae": 854.10,
  "test_samples": 144,
  "lstm_rmse": 1045.84,
  "lstm_mae": 854.34,
  "gru_rmse": 1041.34,
  "gru_mae": 854.10
}
```

### GET /api/predictions
Returns comparison data:
```json
{
  "days": [...],
  "actual": [...],
  "lstm_predictions": [...],
  "gru_predictions": [...]
}
```

### GET /api/rmse-mae-comparison
Returns model comparison metrics

### GET /api/daily-sales
Returns last 90 days of sales data

### POST /api/predict
Makes a prediction given 14-day sequence:
```json
{
  "sequence": [1000, 1100, 1050, ...]
}
```

Response:
```json
{
  "prediction": 1234.56,
  "input_values": [...],
  "status": "success"
}
```

## 📋 Requirements

**Python:** 3.10-slim (as defined in `Dockerfile`)

**Core Dependencies:**
- TensorFlow 2.17.0
- Flask 3.1.0
- NumPy 1.26.4
- Pandas 2.0.3
- Scikit-learn 1.3.1

See `requirements-gcp.txt` for complete list.

## 🛠️ Troubleshooting

### Models not found error
Ensure you've trained and saved the models:
```bash
python preprocessing.py
python train_lstm.py
python train_gru.py
```

### Port already in use (localhost:8080)
Either kill the process or change the `PORT` environment variable in `app.py`

### Deployment fails
Check Cloud Run quotas and ensure service has sufficient permissions:
```bash
gcloud run services describe sales-sense --region europe-west9
```

## 📚 File Descriptions

### `app.py`
Flask application with routes for dashboard and API endpoints. Loads pre-trained models and handles predictions.

### `preprocessing.py`
Loads retail data, normalizes with MinMaxScaler, creates sequences with 14-day sliding window.

### `train_lstm.py` & `train_gru.py`
Model training with validation split, saves trained models to `.h5` format.

### `cross_validation.py`
5-fold walk-forward time-series cross-validation for robust evaluation.

### `compare.py`
Compares LSTM and GRU performance metrics.

### `Dockerfile`
Multi-stage Docker build for production deployment on Google Cloud Run.

### `cloudbuild.yaml`
Google Cloud Build pipeline configuration for CI/CD.

## 📊 Data Information

**Source:** Retail store inventory dataset  
**Period:** 2022-2024  
**Records:** 73,100  
**Time Series Length:** 731 days  
**Target Variable:** Daily sales (units)

## 🔐 Security Notes

- Models and data are not committed to version control (use `.gitignore`)
- API has no authentication (enable if needed for production)
- Consider rate limiting for prediction API

## 📞 Support & Issues

For issues or questions:
1. Check logs: `gcloud run logs read sales-sense --limit 50`
2. Review Flask console output locally
3. Verify model files exist and are readable

## 📄 License

[Specify your license here]

## 👥 Authors

**Ons ELFEKIH** & **Guizani Eya**  
IT Engineering Students — Business Intelligence (Semester 2)
