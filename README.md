# SalesSense: Retail Sales Forecasting with Deep Learning

A deep learning project for predicting daily retail store sales using LSTM and GRU neural networks. Includes data preprocessing, model training, cross-validation evaluation, and a production-ready Streamlit dashboard.

## 🎯 Project Overview

This project implements a time-series forecasting system using a real-world retail store inventory dataset spanning 2022-2024 (731 days, 73,100 records). The goal is to predict daily sales using a 14-day sliding window with two recurrent neural network architectures: LSTM and GRU.

**Best Model:** GRU with a mean RMSE of **1,041.34 units** (5-fold cross-validation)

## 📁 Project Structure

```
SalesSense/
├── dataset/
│   └── retail_store_inventory.csv    # Raw data (73,100 records)
├── preprocessing.py                  # Data loading, normalization, sequence generation
├── train_lstm.py                     # LSTM model training and evaluation
├── train_gru.py                      # GRU model training and evaluation
├── cross_validation.py               # 5-fold walk-forward time-series CV
├── compare.py                        # Model comparison and metrics
├── visualise_predictions.py          # Generate prediction plots
├── streamlit_app.py                  # Interactive dashboard
├── best_model.h5                     # Trained GRU model (production)
├── gru_model.h5, lstm_model.h5       # Saved models
├── scaler.pkl                        # MinMaxScaler for data normalization
├── X_train.npy, X_test.npy           # Preprocessed sequences
├── y_train.npy, y_test.npy           # Target values
├── gru_y_pred.npy, lstm_y_pred.npy   # Model predictions
├── requirements.txt                  # Python dependencies
├── runtime.txt                       # Python version for deployment
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/OnsElfekih/SalesSense.git
cd SalesSense
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python preprocessing.py
```

Outputs:
- `X_train.npy`, `X_test.npy` — Normalized sequences
- `y_train.npy`, `y_test.npy` — Target values
- `scaler.pkl` — MinMaxScaler for inference

### 3. Train Models

```bash
# Train LSTM
python train_lstm.py

# Train GRU
python train_gru.py
```

### 4. Evaluate with Cross-Validation

```bash
python cross_validation.py
```

### 5. Compare Models

```bash
python compare.py
```

### 6. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

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

### Single-Split Performance (test set: 144 days)

| Model | RMSE | MAE | Notes |
|-------|------|-----|-------|
| LSTM | 1,058.01 | 854.34 | Single split baseline |
| GRU | 1,058.54 | 854.10 | Single split baseline |

### 5-Fold Walk-Forward Cross-Validation

| Model | Mean RMSE | Performance |
|-------|-----------|-------------|
| **GRU** | **1,041.34** | ✅ Selected (stable, fewer params) |
| LSTM | 1,045.84 | Marginally higher RMSE |

**Key Findings:**
- Both models capture the overall trend well
- GRU achieves slightly lower RMSE with fewer parameters
- GRU exhibits more stable generalisation across folds
- Peak error occurs at Fold 3 (middle of dataset), then improves as training data grows
- Models struggle with daily fluctuations (univariate data, no external features)
- Error is ~8% relative to mean daily sales (13,200 units)

## 🎨 Dashboard Features

The Streamlit app includes:

- **Overview** — Key metrics and dataset summary
- **EDA** — Exploratory data analysis with visualizations
- **Model Results** — Detailed metrics, comparison charts, residual analysis
- **Predict** — Interactive forecasting with live predictions

## 🌐 Deployment

### Streamlit Cloud

App URL: [salessense-retailsalesforecasting.streamlit.app](https://salessense-retailsalesforecasting.streamlit.app)

**Deployment Steps:**

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Specify `streamlit_app.py` as main file
4. Set Python version in `runtime.txt` (currently 3.11.8)

**Required Files in Repo:**
- ✅ `streamlit_app.py`
- ✅ `best_model.h5` (GRU)
- ✅ `scaler.pkl` (MinMaxScaler)
- ✅ `X_test.npy, y_test.npy`
- ✅ `lstm_y_pred.npy, gru_y_pred.npy`
- ✅ `requirements.txt`
- ✅ `runtime.txt`

## 📋 Requirements

```
tensorflow==2.17.0
scikit-learn==1.3.1
numpy==1.26.4
pandas==2.0.3
matplotlib==3.8.0
streamlit==1.30.0
flask==3.1.0
```

**Python Version:** 3.11.8+

## 📖 Usage Examples

### Make a Single Prediction

```python
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Load model and scaler
model = load_model("best_model.h5", compile=False)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare 14-day sequence (scaled)
sequence = X_test[0].reshape(1, 14, 1)

# Predict
prediction_scaled = model.predict(sequence)
prediction = scaler.inverse_transform([[prediction_scaled[0][0]]])
print(f"Predicted sales: {prediction[0][0]:.0f} units")
```

### Load Cross-Validation Results

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = np.load("y_test.npy")
gru_pred = np.load("gru_y_pred.npy")

rmse = np.sqrt(mean_squared_error(y_true, gru_pred))
mae = mean_absolute_error(y_true, gru_pred)

print(f"GRU RMSE: {rmse:.2f} units")
print(f"GRU MAE: {mae:.2f} units")
```

## 🔍 Key Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Penalizes larger errors; in units sold
- **MAE** (Mean Absolute Error): Average absolute difference; easier to interpret
- **R²**: Proportion of variance explained (0-1 scale)

Lower RMSE/MAE is better. Higher R² is better.

## 🛠 Troubleshooting

### `scaler.pkl` missing error
```bash
python preprocessing.py
```

### TensorFlow compatibility issues
Ensure Python 3.11+ and TensorFlow >= 2.17:
```bash
pip install --upgrade tensorflow==2.17.0
```

### Model loads but predictions are wrong
Verify `scaler.pkl` matches the training data. Retrain if necessary:
```bash
python preprocessing.py
python train_gru.py
```

## 📚 Report Integration

This project includes visualizations and results for academic/technical reports:

- `images/cv_results.png` — 5-fold CV comparison chart
- `images/prediction_full.png` — Full test period predictions
- `images/prediction_zoom.png` — Zoomed-in view of predictions
- `images/prediction_residuals.png` — Residual analysis
- `images/rolling_avg_trend.png` — 7-day rolling average trend

## 👩‍💻 Authors

**Ons ELFEKIH**  **Guizani Eya**
IT Engineering Students — Business Intelligence  