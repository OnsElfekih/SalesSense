"""
cross_validation.py
-------------------
Walk-forward time series cross-validation using sklearn TimeSeriesSplit.
5 folds — metrics reported in original units (units sold).
Fixed seed 42 for reproducibility.
Run: python cross_validation.py
"""

import os, pickle, random
import numpy as np

# ── Seed BEFORE any TensorFlow import ────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.random.set_seed(SEED)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

print("\n=== TIME-SERIES CROSS-VALIDATION FOR SALES FORECASTING ===")
print(f"Random seed: {SEED}  (reproducible)")

DATA_FILES = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "scaler.pkl"]
if not all(os.path.exists(f) for f in DATA_FILES):
    print("\n[ERROR] Required files not found. Run compare.py first.")
else:
    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test  = np.load("y_test.npy")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_combined = np.concatenate([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, y_test], axis=0)

    print(f"\nCombined dataset: X={X_combined.shape}, y={y_combined.shape}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    tscv       = TimeSeriesSplit(n_splits=5)
    SEQ_LEN    = 14
    EPOCHS     = 30
    BATCH_SIZE = 32

    cv_results = {"fold": [], "lstm_rmse": [], "lstm_mae": [],
                  "gru_rmse": [], "gru_mae": []}

    print(f"\n5-fold walk-forward CV  |  epochs/fold: {EPOCHS}  |  metrics: raw units\n")

    # ── Model builders ────────────────────────────────────────────────────────
    def build_lstm():
        tf.random.set_seed(SEED)
        m = Sequential([Input(shape=(SEQ_LEN, 1)),
                        LSTM(64, return_sequences=True), Dropout(0.2),
                        LSTM(32), Dropout(0.2), Dense(1)])
        m.compile(optimizer=Adam(1e-3), loss="mse")
        return m

    def build_gru():
        tf.random.set_seed(SEED)
        m = Sequential([Input(shape=(SEQ_LEN, 1)),
                        GRU(64, return_sequences=True), Dropout(0.2),
                        GRU(32), Dropout(0.2), Dense(1)])
        m.compile(optimizer=Adam(1e-3), loss="mse")
        return m

    def inverse(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    # ── CV loop ───────────────────────────────────────────────────────────────
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined), 1):
        X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
        y_tr, y_val = y_combined[train_idx], y_combined[val_idx]
        y_val_raw   = inverse(y_val)

        print(f"Fold {fold}/5 — Train: {len(train_idx)} | Val: {len(val_idx)}")

        # LSTM — predict once, store result
        lstm_m = build_lstm()
        lstm_m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_split=0.1, verbose=0)
        lstm_pred_raw = inverse(lstm_m.predict(X_val, verbose=0).flatten())
        lstm_rmse = float(np.sqrt(mean_squared_error(y_val_raw, lstm_pred_raw)))
        lstm_mae  = float(mean_absolute_error(y_val_raw, lstm_pred_raw))

        # GRU — predict once, store result
        gru_m = build_gru()
        gru_m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, verbose=0)
        gru_pred_raw = inverse(gru_m.predict(X_val, verbose=0).flatten())
        gru_rmse = float(np.sqrt(mean_squared_error(y_val_raw, gru_pred_raw)))
        gru_mae  = float(mean_absolute_error(y_val_raw, gru_pred_raw))

        cv_results["fold"].append(f"Fold {fold}")
        cv_results["lstm_rmse"].append(lstm_rmse)
        cv_results["lstm_mae"].append(lstm_mae)
        cv_results["gru_rmse"].append(gru_rmse)
        cv_results["gru_mae"].append(gru_mae)

        print(f"  LSTM: RMSE={lstm_rmse:.2f} units  MAE={lstm_mae:.2f} units")
        print(f"  GRU:  RMSE={gru_rmse:.2f} units  MAE={gru_mae:.2f} units\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_results)

    print("=" * 70)
    print("CROSS-VALIDATION RESULTS SUMMARY  (units sold)")
    print("=" * 70)
    print(cv_df.to_string(index=False))

    lstm_avg_rmse = np.mean(cv_results["lstm_rmse"])
    gru_avg_rmse  = np.mean(cv_results["gru_rmse"])
    lstm_avg_mae  = np.mean(cv_results["lstm_mae"])
    gru_avg_mae   = np.mean(cv_results["gru_mae"])

    print(f"\nLSTM — Mean RMSE: {lstm_avg_rmse:.2f} ± {np.std(cv_results['lstm_rmse']):.2f} units"
          f"  |  Mean MAE: {lstm_avg_mae:.2f} ± {np.std(cv_results['lstm_mae']):.2f} units")
    print(f"GRU  — Mean RMSE: {gru_avg_rmse:.2f} ± {np.std(cv_results['gru_rmse']):.2f} units"
          f"  |  Mean MAE: {gru_avg_mae:.2f} ± {np.std(cv_results['gru_mae']):.2f} units")

    winner = "LSTM" if lstm_avg_rmse < gru_avg_rmse else "GRU"
    print(f"\nWINNER: {winner}  (lower mean CV RMSE in original units)")
    print("=" * 70)

    # ── Plot ──────────────────────────────────────────────────────────────────
    x_pos = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, lk, gk, lavg, gavg, ylabel, title in [
        (axes[0], "lstm_rmse", "gru_rmse", lstm_avg_rmse, gru_avg_rmse,
         "RMSE (units sold)", "RMSE across folds"),
        (axes[1], "lstm_mae",  "gru_mae",  lstm_avg_mae,  gru_avg_mae,
         "MAE (units sold)",  "MAE across folds"),
    ]:
        ax.plot(x_pos, cv_results[lk], marker="o", label="LSTM", linewidth=2)
        ax.plot(x_pos, cv_results[gk], marker="s", label="GRU",  linewidth=2)
        ax.axhline(lavg, color="steelblue",  linestyle="--", alpha=0.6,
                   label=f"LSTM avg: {lavg:.2f}")
        ax.axhline(gavg, color="darkorange", linestyle="--", alpha=0.6,
                   label=f"GRU avg:  {gavg:.2f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cv_results["fold"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("5-Fold Walk-Forward Cross-Validation — LSTM vs GRU", fontweight="bold")
    plt.tight_layout()
    plt.savefig("cv_results.png", dpi=130, bbox_inches="tight")
    print("\nChart saved → cv_results.png")
    print("Cross-validation complete.")