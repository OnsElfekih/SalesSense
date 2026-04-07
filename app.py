"""
Flask web app for SalesSense — Sales Forecasting Dashboard
Deploy to Google Cloud Run
Run locally: python app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SEQ_LEN = 14
PORT = int(os.environ.get("PORT", 8080))

def find_file(filename):
    """Search for file in root or dataset subfolder."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "dataset" / filename,
        BASE_DIR / "data" / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(BASE_DIR / filename)

# ── Load artifacts (cached on startup) ─────────────────────────────────────────
print("[Loading] Model and data artifacts...")
artifacts = {}
try:
    artifacts['model'] = load_model(find_file("best_model.h5"), compile=False)
    with open(find_file("scaler.pkl"), "rb") as f:
        artifacts['scaler'] = pickle.load(f)
    
    artifacts['X_test'] = np.load(find_file("X_test.npy"))
    artifacts['y_true'] = np.load(find_file("y_true.npy"))
    artifacts['lstm_pred'] = np.load(find_file("lstm_y_pred.npy"))
    artifacts['gru_pred'] = np.load(find_file("gru_y_pred.npy"))
    
    df = pd.read_csv(find_file("retail_store_inventory.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    artifacts['daily'] = (df.groupby("Date")["Units Sold"].sum()
             .reset_index().sort_values("Date")
             .rename(columns={"Units Sold": "sales"}))
    
    print("[OK] All artifacts loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load artifacts: {e}")

# ── Calculate metrics ──────────────────────────────────────────────────────────
if 'y_true' in artifacts and 'lstm_pred' in artifacts:
    lstm_rmse = float(np.sqrt(mean_squared_error(artifacts['y_true'], artifacts['lstm_pred'])))
    lstm_mae = float(mean_absolute_error(artifacts['y_true'], artifacts['lstm_pred']))
    gru_rmse = float(np.sqrt(mean_squared_error(artifacts['y_true'], artifacts['gru_pred'])))
    gru_mae = float(mean_absolute_error(artifacts['y_true'], artifacts['gru_pred']))
else:
    lstm_rmse = lstm_mae = gru_rmse = gru_mae = 0

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")

@app.route("/api/overview")
def api_overview():
    """Get overview metrics."""
    return jsonify({
        "best_model": "GRU",
        "gru_cv_rmse": 1041.34,
        "gru_test_mae": gru_mae,
        "test_samples": 144,
        "lstm_rmse": lstm_rmse,
        "lstm_mae": lstm_mae,
        "gru_rmse": gru_rmse,
        "gru_mae": gru_mae
    })

@app.route("/api/predictions")
def api_predictions():
    """Get model predictions for plotting."""
    if 'lstm_pred' not in artifacts or 'gru_pred' not in artifacts:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = {
        "lstm_predictions": artifacts['lstm_pred'][:50].tolist(),
        "gru_predictions": artifacts['gru_pred'][:50].tolist(),
        "actual": artifacts['y_true'][:50].tolist(),
        "days": list(range(1, 51))
    }
    return jsonify(data)

@app.route("/api/rmse-mae-comparison")
def api_rmse_mae():
    """Get RMSE/MAE comparison chart data."""
    return jsonify({
        "models": ["LSTM", "GRU"],
        "rmse": [lstm_rmse, gru_rmse],
        "mae": [lstm_mae, gru_mae]
    })

@app.route("/api/daily-sales")
def api_daily_sales():
    """Get last 90 days predicted sales time series."""
    if 'gru_pred' not in artifacts:
        return jsonify({"error": "Predictions not loaded"}), 500
    
    predictions = artifacts['gru_pred'][-90:]
    dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq='D')
    
    data = {
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "sales": predictions.tolist()
    }
    return jsonify(data)

@app.route("/api/predict", methods=["POST"])
def predict():
    """Make a prediction with custom input."""
    if 'model' not in artifacts or 'scaler' not in artifacts:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        sequence = np.array(data.get("sequence", []), dtype=float)
        
        # Validate input length
        if len(sequence) != SEQ_LEN:
            return jsonify({"error": f"Expected {SEQ_LEN} values, got {len(sequence)}"}), 400
        
        # Normalize using the scaler
        sequence_normalized = artifacts['scaler'].transform(sequence.reshape(-1, 1)).flatten()
        sequence_reshaped = sequence_normalized.reshape(1, SEQ_LEN, 1)
        
        # Make prediction
        prediction_normalized = artifacts['model'].predict(sequence_reshaped, verbose=0)[0][0]
        
        # Denormalize
        prediction = artifacts['scaler'].inverse_transform([[prediction_normalized]])[0][0]
        
        return jsonify({
            "prediction": float(prediction),
            "input_values": sequence.tolist(),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    """Health check endpoint for Cloud Run."""
    return jsonify({"status": "healthy"}), 200

# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cloud Run sets PORT env variable
    app.run(host="0.0.0.0", port=PORT, debug=False)
